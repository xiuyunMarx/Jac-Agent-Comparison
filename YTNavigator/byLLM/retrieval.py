"""Retrieval plumbing for the byLLM YT-Navigator counterpart.

Framework-neutral Python: hits the same Postgres/PGVector tables the LangGraph
implementation populates (app_channel / app_video / app_videochunk /
langchain_pg_*), so both agents answer over identical data. Kept out of the
.jac files on purpose - the comparison is about the agent frameworks, not the
database plumbing.

Parity notes vs. the original tools:
- Semantic search: same embedding model (BAAI/bge-small-en-v1.5), same top-20
  cosine search against langchain_pg_embedding, filtered to the channel's
  collection.
- Keyword search: BM25 (Okapi, k1=1.5 b=0.75) over the channel's chunks, top 4
  - a dependency-free reimplementation of the original's BM25Retriever.
- Merge, dedup by text, keep the 5 most-cited videos, score-sorted - mirroring
  the original tool's grouping. The cross-encoder reranker is NOT reproduced
  (documented divergence; both searches share the embedding space regardless).
- SQL tool: SELECT-only, restricted to the video/chunk tables, 20-row cap,
  schema echoed back on errors - same contract as the original.

Requires: psycopg2-binary, sentence-transformers (lazy-loaded on first search).
Reads POSTGRES_* env vars (same names as YT-Navigator's .env).
"""

import hashlib
import math
import os
import random
import re
from collections import Counter

_EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = 384
_embedder = None

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def fake_embedding(text):
    """Deterministic pseudo-embedding used when YTNAV_FAKE_EMBEDDINGS is set.

    Lets the whole pipeline run without torch (smoke tests); retrieval quality
    is meaningless. datasets/build.py imports this so stored and query vectors
    come from the same function.
    """
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(EMBEDDING_DIM)]


def _fake_mode():
    return os.environ.get("YTNAV_FAKE_EMBEDDINGS", "").lower() in ("1", "true", "yes")


def _dsn():
    """Build a Postgres DSN from the same env vars YT-Navigator uses."""
    return (
        f"postgresql://{os.environ.get('POSTGRES_USER', 'postgres')}:"
        f"{os.environ.get('POSTGRES_PASSWORD', '')}@"
        f"{os.environ.get('POSTGRES_HOST', 'localhost')}:"
        f"{os.environ.get('POSTGRES_PORT', '5432')}/"
        f"{os.environ.get('POSTGRES_DB', 'postgres')}"
    )


def _connect():
    import psycopg2

    return psycopg2.connect(_dsn())


def _embed(text):
    """Embed a query with the same model that populated the vector store."""
    global _embedder
    if _fake_mode():
        return fake_embedding(text)
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers is required for semantic search: pip install sentence-transformers"
            ) from e
        _embedder = SentenceTransformer(_EMBEDDING_MODEL_NAME)
    return _embedder.encode(text).tolist()


def resolve_channel(preferred=""):
    """Return the channel id to benchmark against.

    Uses `preferred` when given, otherwise the only channel in the database.
    Raises RuntimeError when no channel or several channels exist.
    """
    with _connect() as conn, conn.cursor() as cur:
        if preferred:
            cur.execute("SELECT id FROM app_channel WHERE id = %s", (preferred,))
            row = cur.fetchone()
            if row is None:
                cur.execute("SELECT id FROM app_channel")
                raise RuntimeError(
                    f"Channel '{preferred}' not found. Available: {[r[0] for r in cur.fetchall()] or 'none'}"
                )
            return row[0]
        cur.execute("SELECT id FROM app_channel")
        rows = cur.fetchall()
        if not rows:
            raise RuntimeError("No channels in the database - load a snapshot or scan a channel first")
        if len(rows) > 1:
            raise RuntimeError(f"Multiple channels found, set YTNAV_CHANNEL. Available: {[r[0] for r in rows]}")
        return rows[0][0]


def channel_info(channel_id):
    """Pretty channel description - mirrors Channel.pretty_str() in the original."""
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT c.id, c.name, c.username, c.description, "
            "(SELECT COUNT(*) FROM app_video v WHERE v.channel_id = c.id) "
            "FROM app_channel c WHERE c.id = %s",
            (channel_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise RuntimeError(f"Channel '{channel_id}' not found")
        return (
            f"\n        ID: {row[0]}\n        Name: {row[1]}\n        Username: {row[2]}\n"
            f"        Description: {row[3]}\n        Scanned Videos Count: {row[4]}\n        "
        )


def _tokenize(text):
    return _TOKEN_RE.findall(text.lower())


def _bm25_search(cur, query, channel_id, k=4, k1=1.5, b=0.75):
    """Okapi BM25 over the channel's chunks; returns [(text, video_id, start, end)]."""
    cur.execute(
        "SELECT vc.text, vc.video_id, vc.start, vc.\"end\" "
        "FROM app_videochunk vc JOIN app_video v ON vc.video_id = v.id "
        "WHERE v.channel_id = %s",
        (channel_id,),
    )
    rows = cur.fetchall()
    if not rows:
        return []

    corpus = [_tokenize(r[0]) for r in rows]
    doc_lens = [len(d) for d in corpus]
    avgdl = sum(doc_lens) / len(doc_lens) if doc_lens else 0
    doc_freq = Counter()
    for doc in corpus:
        doc_freq.update(set(doc))

    n = len(corpus)
    query_terms = _tokenize(query)
    scores = []
    for i, doc in enumerate(corpus):
        tf = Counter(doc)
        score = 0.0
        for term in query_terms:
            if term not in tf:
                continue
            idf = math.log(1 + (n - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            denominator = tf[term] + k1 * (1 - b + b * doc_lens[i] / avgdl) if avgdl else tf[term] + k1
            score += idf * (tf[term] * (k1 + 1)) / denominator
        scores.append((score, i))

    scores.sort(reverse=True)
    return [rows[i] for score, i in scores[:k] if score > 0]


def _semantic_search(cur, query, channel_id, k=20):
    """Cosine top-k against the channel's PGVector collection."""
    vector_literal = "[" + ",".join(f"{x:.8f}" for x in _embed(query)) + "]"
    cur.execute(
        "SELECT e.document, e.cmetadata, 1 - (e.embedding <=> %s::vector) AS similarity "
        "FROM langchain_pg_embedding e "
        "JOIN langchain_pg_collection c ON e.collection_id = c.uuid "
        "WHERE c.name = %s "
        "ORDER BY e.embedding <=> %s::vector ASC LIMIT %s",
        (vector_literal, channel_id, vector_literal, k),
    )
    return cur.fetchall()


def search_videos_impl(query, channel_id):
    """Hybrid search over the channel's transcripts; returns an LLM-readable report.

    Combines semantic (top 20) and BM25 keyword (top 4) hits, deduplicates by
    text, keeps chunks from the 5 most-cited videos, and formats them grouped
    by video with titles, thumbnails, and timestamps.
    """
    with _connect() as conn, conn.cursor() as cur:
        chunks = []  # (text, video_id, start, end, score)
        for document, metadata, similarity in _semantic_search(cur, query, channel_id):
            metadata = metadata or {}
            start = metadata.get("start_time")
            duration = metadata.get("duration") or 0
            end = (start + duration) if isinstance(start, (int, float)) else None
            chunks.append((document, metadata.get("video_id"), start, end, round(float(similarity) * 100, 1)))
        for text, video_id, start, end in _bm25_search(cur, query, channel_id):
            chunks.append((text, video_id, str(start) if start else None, str(end) if end else None, 50.0))

        seen, deduped = set(), []
        for chunk in chunks:
            if chunk[0] not in seen:
                seen.add(chunk[0])
                deduped.append(chunk)
        deduped = [c for c in deduped if c[1]]
        if not deduped:
            return "No results found for this query."

        top_video_ids = [vid for vid, _ in Counter(c[1] for c in deduped).most_common(5)]
        cur.execute(
            "SELECT id, title, thumbnail, published_at FROM app_video WHERE id = ANY(%s)",
            (top_video_ids,),
        )
        videos = {row[0]: row for row in cur.fetchall()}

    lines = []
    for vid in top_video_ids:
        if vid not in videos:
            continue
        _, title, thumbnail, published_at = videos[vid]
        lines.append(f"## Video: {title}\n- id: {vid}\n- thumbnail_url: {thumbnail}\n- published_at: {published_at}")
        for text, video_id, start, end, score in deduped:
            if video_id != vid:
                continue
            lines.append(f"  - [start={start} end={end} relevance={score}] {text}")
    return "\n".join(lines) if lines else "No results found for this query."


_SCHEMA_MARKDOWN = """### app_video
| Field | Type |
|-------|------|
| id | CharField |
| title | CharField |
| thumbnail | URLField |
| published_at | DateTimeField |
| channel | ForeignKey (column: channel_id) |

### app_videochunk
| Field | Type |
|-------|------|
| id | BigAutoField |
| video | ForeignKey (column: video_id) |
| start | TimeField |
| end | TimeField |
| text | TextField |
"""


def run_sql_impl(query):
    """Execute a SELECT over the video/chunk tables - same contract as the original tool."""
    query = query.strip()
    if not query.startswith("SELECT"):
        return "Error: Only SELECT queries are supported"
    if "app_video" not in query and "app_videochunk" not in query:
        return (
            f"DB SCHEMA:\n{_SCHEMA_MARKDOWN}\n"
            "Error: You are allowed only to search in the app_video and app_videochunk tables"
        )
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(query)
            columns = [d[0] for d in cur.description]
            rows = [str(dict(zip(columns, row))) for row in cur.fetchall()]
        if len(rows) > 20:
            return "\n".join(rows[:20]) + (
                f"\nThe result is too long; truncated to 20 rows from a total of {len(rows)} rows."
            )
        return "\n".join(rows) if rows else "(no rows)"
    except Exception as e:
        return f"DB SCHEMA:\n{_SCHEMA_MARKDOWN}\n#Error: {e}"


def sql_schema_markdown():
    """Schema text advertised to the LLM in the SQL tool description."""
    return _SCHEMA_MARKDOWN


def duration_seconds(start_time, end_time):
    """Seconds between two litellm callback timestamps, or None."""
    try:
        return round((end_time - start_time).total_seconds(), 4)
    except Exception:
        return None
