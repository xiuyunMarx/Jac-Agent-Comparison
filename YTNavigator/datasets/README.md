# Benchmark dataset

A fully synthetic, fully reproducible YouTube channel — no scraping, no rate
limits, no content drift. Both agent implementations run against exactly this
data.

- **`channel_data.json`** — the "NeuralBytes" channel: 10 ML-explainer videos
  ("Transformers Explained", "GANs", "Fine-Tuning LLMs on a Budget", ...) with
  68 timestamped transcript chunks written so each video has distinctive
  vocabulary (retrieval questions have unambiguous correct answers).
- **`questions.jsonl`** — 23 questions with ground truth: 18 tool-route
  ("Yes"), 3 direct ("No"), 2 refusal ("Not relevant"); 16 with
  `expected_video_ids`, all 23 with `reference_answer` for LLM-judge scoring,
  and two multi-turn scenarios (`s1`, `s2`).
- **`build.py`** — loads the channel into Postgres: `app_channel` /
  `app_video` / `app_videochunk` plus the PGVector tables
  (`langchain_pg_collection` / `langchain_pg_embedding`), creating them if
  missing (schemas compatible with the Django app — `manage.py migrate
  --fake-initial` accepts them — and with langchain_postgres). Embeddings use
  the same model as the original app (`BAAI/bge-small-en-v1.5`).

```bash
python build.py               # idempotent; --replace to rebuild
python build.py --fake-embeddings   # smoke mode, no torch (retrieval meaningless)
```

`--fake-embeddings` pairs with `YTNAV_FAKE_EMBEDDINGS=1` on the query side
(set automatically by `eval/e2e.py --fake-embeddings`) so stored and query
vectors come from the same function.

Ground-truth invariants the questions rely on: 10 videos total; most recent =
"Fine-Tuning LLMs on a Budget" (2025-03-15); exactly two 2025 videos; the
facts in each `reference_answer` appear verbatim-ish in the transcript chunks.
If you edit `channel_data.json`, re-check `questions.jsonl`.
