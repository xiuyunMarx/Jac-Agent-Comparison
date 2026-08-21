"""RAG engine for Jac-GPT, no-framework port: FAISS vector search over the
bundled docs with CrossEncoder reranking — `faiss`, `sentence-transformers`
and `pickle` directly, no langchain.

The index under faiss_index/ is a byte-copy of ../langgraph's, so both sides
search the exact same vectors over the exact same chunks. That index is in
langchain's save_local format: a raw faiss file plus a pickle of
(InMemoryDocstore, index_to_docstore_id). The pickle references two langchain
classes; a restricted unpickler maps exactly those two onto local stand-ins
and refuses everything else, so loading it needs no langchain import and
cannot execute anything the docstore did not put there.

What langchain's FAISS wrapper was doing at query time is four lines: embed
the query with the same local sentence-transformers model that built the
index (BAAI/bge-small-en-v1.5 — no API key involved), `index.search` for the
top chunk_k L2 neighbours, map row ids through index_to_docstore_id, look the
documents up in the docstore. Reranking and the summary string were always
plain Python on every side; they are ported line-for-line from
../langgraph/rag_engine.py, CrossEncoder arguments included.

This side does not rebuild the index (that would mean reimplementing
langchain's RecursiveCharacterTextSplitter and loaders, and any drift there
would silently change the corpus). A missing index is a setup error and
raises, pointing at the langgraph side's builder.
"""

from dataclasses import dataclass
import json
import os
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer


@dataclass
class Config:
    """Defaults mirror config/faiss_reranking.json, as on the other sides."""
    model_name: str = "gpt-4.1-mini"
    chunk_size: int = 800
    chunk_overlap: int = 100
    chunk_k: int = 30  # approximate top-k search param
    rerank_top_n: int = 7  # rerank top-n
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    embedding_model: str = "BAAI/bge-small-en-v1.5"  # local sentence-transformers model
    docs_path: str = "docs"
    faiss_path: str = "faiss_index"


def load_config(config_path: str) -> Config:
    """Parse the nested config/faiss_reranking.json into a flat Config.

    The Jac sides parse this nested layout properly; the LangGraph side
    passes the nested dict to a flat dataclass, always fails, and lands on
    defaults that happen to equal the file (a known, documented asymmetry).
    This side parses it properly — same effective values either way.
    """
    cfg = Config()
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        llm_cfg = raw.get("llm", {})
        rag_cfg = raw.get("rag_engine", {})
        search_cfg = rag_cfg.get("similarity_search", {})
        rerank_cfg = rag_cfg.get("reranking", {})
        paths_cfg = raw.get("paths", {})
        cfg.model_name = llm_cfg.get("model_name", cfg.model_name)
        cfg.chunk_size = rag_cfg.get("chunk_size", cfg.chunk_size)
        cfg.chunk_overlap = rag_cfg.get("chunk_overlap", cfg.chunk_overlap)
        cfg.chunk_k = search_cfg.get("k", cfg.chunk_k)
        cfg.rerank_top_n = rerank_cfg.get("top_n", cfg.rerank_top_n)
        cfg.cross_encoder_model = rerank_cfg.get("model", cfg.cross_encoder_model)
        cfg.embedding_model = rag_cfg.get("embedding_model", cfg.embedding_model)
        cfg.docs_path = paths_cfg.get("file_path", cfg.docs_path)
        cfg.faiss_path = paths_cfg.get("faiss_path", cfg.faiss_path)
    except Exception as e:
        print(f"[config] Could not load {config_path} ({e}); using built-in defaults.")
    return cfg


class Document:
    """Stand-in for langchain_core.documents.base.Document.

    Pydantic v2 models pickle as new-empty-object + __setstate__ with a dict
    holding the real attributes under '__dict__'; capturing that is all a
    document is here: page_content plus metadata (source, id,
    original_content, and page for PDFs).
    """
    page_content: str
    metadata: Dict[str, Any]

    def __setstate__(self, state: Any) -> None:
        inner = state.get("__dict__", state) if isinstance(state, dict) else {}
        self.page_content = inner.get("page_content", "")
        self.metadata = inner.get("metadata", {}) or {}


class _Docstore:
    """Stand-in for langchain_community.docstore.in_memory.InMemoryDocstore."""
    _dict: Dict[str, Document]

    def __setstate__(self, state: Any) -> None:
        self._dict = state.get("_dict", {}) if isinstance(state, dict) else {}


_ALLOWED_CLASSES = {
    ("langchain_community.docstore.in_memory", "InMemoryDocstore"): _Docstore,
    ("langchain_core.documents.base", "Document"): Document,
}


class _IndexUnpickler(pickle.Unpickler):
    """Unpickler that admits only the two classes index.pkl legitimately holds."""

    def find_class(self, module: str, name: str) -> Any:
        try:
            return _ALLOWED_CLASSES[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError(
                f"index.pkl references unexpected class {module}.{name}; "
                "refusing to unpickle it")


class RagEngine:
    def __init__(self, config_path: str) -> None:
        self.config: Config = load_config(config_path)
        self.cross_encoder: Optional[CrossEncoder] = None
        try:
            self.cross_encoder = CrossEncoder(self.config.cross_encoder_model)
        except Exception:
            print(f"CrossEncoder model {self.config.cross_encoder_model} not found, reranking disabled. ")

        print("RAG Engine initialized")
        print(f"Chunk size: {self.config.chunk_size}, Overlap: {self.config.chunk_overlap}, "
              f"Chunk Nos: {self.config.chunk_k}")

        # Config paths are relative to this module, not to the caller's cwd.
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.docs_path: str = os.path.join(base_dir, self.config.docs_path)
        self.faiss_path: str = os.path.join(base_dir, self.config.faiss_path)

        # Embeddings run locally: searching the index needs no API key.
        self.embedder: Optional[SentenceTransformer] = None
        try:
            self.embedder = SentenceTransformer(self.config.embedding_model)
            print(f"Local embeddings enabled with model: {self.config.embedding_model}")
        except Exception as e:
            print(f"Failed to load embedding model {self.config.embedding_model}: {e}. RAG search disabled.")

        self.index: Optional[faiss.Index] = None
        self.docstore: Dict[str, Document] = {}
        self.index_to_docstore_id: Dict[int, str] = {}
        if self.embedder is not None:
            self._load_index()

    def _load_index(self) -> None:
        index_file = os.path.join(self.faiss_path, "index.faiss")
        pkl_file = os.path.join(self.faiss_path, "index.pkl")
        if not (os.path.isfile(index_file) and os.path.isfile(pkl_file)):
            raise FileNotFoundError(
                f"No FAISS index at {self.faiss_path}. This side does not rebuild "
                "the index; copy index.faiss/index.pkl from ../langgraph/faiss_index "
                "(or rebuild there and copy) so all systems search identical vectors.")
        print(f"Loading existing FAISS index from {self.faiss_path}")
        self.index = faiss.read_index(index_file)
        with open(pkl_file, "rb") as f:
            docstore, self.index_to_docstore_id = _IndexUnpickler(f).load()
        self.docstore = docstore._dict
        if self.index.ntotal != len(self.index_to_docstore_id):
            raise ValueError(
                f"index.faiss holds {self.index.ntotal} vectors but index.pkl maps "
                f"{len(self.index_to_docstore_id)} — the two files are not a pair.")

    def similarity_search_with_score(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """Embed the query and return the k nearest (document, L2 distance) pairs."""
        assert self.embedder is not None and self.index is not None
        embedding = self.embedder.encode([query], show_progress_bar=False)
        vector = np.asarray(embedding, dtype=np.float32)
        scores, indices = self.index.search(vector, k)
        results: List[Tuple[Document, float]] = []
        for score, row in zip(scores[0], indices[0]):
            if row == -1:  # fewer vectors than k
                continue
            results.append((self.docstore[self.index_to_docstore_id[int(row)]], float(score)))
        return results

    def search(self, query: str) -> str:
        """Retrieve chunks for `query`, rerank them, and format them as one summary string."""
        if self.index is None:
            print("FAISS index not initialized")
            return ""

        print(f"Searching RAG Engine with query: {query}")
        rag_start = time.perf_counter()

        search_start = time.perf_counter()
        results = self.similarity_search_with_score(query, k=self.config.chunk_k)
        print(f"Search time: {time.perf_counter() - search_start:.4f}s")

        cross_encoder = self.cross_encoder
        if cross_encoder is not None and results:
            try:
                rerank_start = time.perf_counter()
                docs = [doc for doc, _ in results]
                # Score against the pre-enrichment text, not the metadata-prefixed version.
                scores = cross_encoder.predict(
                    [(query, doc.metadata.get("original_content", doc.page_content)) for doc in docs],
                    show_progress_bar=False,
                    batch_size=self.config.chunk_k,
                    convert_to_numpy=True,
                    convert_to_tensor=False,
                )
                results = sorted(zip(docs, scores), key=lambda pair: -pair[1])
                print(f"Reranking time: {time.perf_counter() - rerank_start:.4f}s")
                print(f"Reranked {len(results)} documents using CrossEncoder")
            except Exception as e:
                print(f"Error during reranking: {e}")
                print("Falling back to original FAISS results")
        results = results[: self.config.rerank_top_n]

        print(f"\n{'='*60}")
        print(f"Retrieved {len(results)} chunks for query: {query}")
        print(f"{'='*60}")

        summary = ""
        for idx, (doc, score) in enumerate(results, 1):
            page = doc.metadata.get("page")
            source = doc.metadata.get("source")
            print(f"\nChunk {idx}:")
            print(f"  Score: {score:.4f}")
            print(f"  Source: {source}")
            print(f"  ID: {doc.metadata.get('id')}")
            if page is not None:
                print(f"  Page: {page}")
            print(f"  Content preview: {doc.page_content[:100]}...")
            summary += f"{source} (Relevance Score: {score:.4f}) : {doc.metadata.get('original_content', doc.page_content)}\n"

        print(f"\n{'='*60}\n")
        print(f"RAG time: {time.perf_counter() - rag_start:.4f}s")
        return summary

    # The LangGraph side exposes retrieval as `rag_engine(query)`; keep both spellings.
    __call__ = search
