"""Score raw eval runs: routing, jac-check compilation, and gpt-4.1 judging.

Reads results/raw_runs.jsonl (last occurrence per system/item/repeat/turn wins)
plus results/proxy_log.jsonl for token joins, writes results/judged.jsonl.

Judge calls go DIRECT to api.openai.com (never through the eval proxy) and are
cached in results/judge_cache.jsonl by (kind, item, response) so repeats with
identical text are judged once.

Run:  /home/xiaoyu/miniconda3/envs/jaseci/bin/python score.py
"""

import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dataset"))
from synthesize import POOL, split_sections  # noqa: E402

JUDGE_CACHE_PATH = common.RESULTS_DIR / "judge_cache.jsonl"
CODE_AGENTS = {"CodingChat", "DebuggerChat"}
WORKERS = 8

_lock = threading.Lock()


class AnswerJudge(BaseModel):
    score: int  # 1-5 vs gold
    acceptable: bool
    reason: str


class CodeJudge(BaseModel):
    score: int  # 1-5 task fulfillment
    reason: str


class BinaryJudge(BaseModel):
    appropriate: bool
    reason: str


common.require_api_key()
client = OpenAI(base_url="https://api.openai.com/v1")
_parse = getattr(client.chat.completions, "parse", None) or client.beta.chat.completions.parse

_cache: dict[str, dict] = {}


def cache_key(kind: str, item_id: str, response: str) -> str:
    return hashlib.md5(f"{kind}|{item_id}|{response}".encode()).hexdigest()


def load_cache() -> None:
    for row in common.read_jsonl(JUDGE_CACHE_PATH):
        _cache[row["key"]] = row["result"]


def judged_llm(key: str, messages: list[dict], schema: type[BaseModel]) -> dict | None:
    with _lock:
        if key in _cache:
            return _cache[key]
    resp = _parse(model=common.JUDGE_MODEL, messages=messages,
                  response_format=schema, temperature=0.0)
    parsed = resp.choices[0].message.parsed
    result = parsed.model_dump() if parsed else None
    if result is not None:
        with _lock:
            _cache[key] = result
            common.append_jsonl(JUDGE_CACHE_PATH, {"key": key, "result": result})
    return result


def build_section_map() -> dict[tuple, str]:
    out = {}
    for rel in POOL:
        try:
            for sec in split_sections(rel):
                out[(sec["source_file"], sec["source_section"])] = sec["content"]
        except FileNotFoundError:
            pass
    return out


def judge_rag(item: dict, response: str, section: str) -> dict | None:
    return judged_llm(cache_key("rag", item["id"], response), [
        {"role": "system", "content":
            "You grade an assistant's answer about the Jac programming language against a "
            "gold answer derived from the official documentation (the excerpt is provided as "
            "ground truth). Score 1-5: 5 = fully correct and complete vs the gold answer; "
            "4 = correct with minor omissions; 3 = partially correct; 2 = mostly wrong or "
            "seriously incomplete; 1 = wrong, irrelevant, or refuses. acceptable = score >= 4. "
            "Extra correct detail beyond the gold answer must not be penalized."},
        {"role": "user", "content":
            f"Documentation excerpt (ground truth):\n{section[:2500]}\n\n"
            f"Question: {item['turns'][0]['message']}\n\n"
            f"Gold answer: {item['gold_answer']}\n\n"
            f"Assistant answer:\n{response[:4000]}"},
    ], AnswerJudge)


def judge_code(item: dict, response: str, compiles: bool | None) -> dict | None:
    if item["category"] == "debugging" or item.get("buggy_code"):
        task_desc = (f"The user submitted this buggy Jac code:\n```jac\n{item['buggy_code']}\n```\n"
                     f"The actual injected bug: {item['bug_description']}\n"
                     f"A correct fixed program:\n```jac\n{item['reference_code']}\n```")
        rubric = ("Score 1-5 on whether the assistant correctly identified and fixed THE bug: "
                  "5 = bug identified and fix is correct; 3 = plausible fix but misses or only "
                  "partially fixes the actual bug; 1 = wrong diagnosis or no usable fix.")
    else:
        task_desc = (f"Task given to the assistant: {item['turns'][0]['message']}\n"
                     f"One valid reference solution:\n```jac\n{item['reference_code']}\n```")
        rubric = ("Score 1-5 on task fulfillment: 5 = complete, correct Jac program satisfying "
                  "the task (it need NOT match the reference — any valid approach counts); "
                  "3 = right idea, notable gaps or errors; 1 = no usable code or wrong language.")
    compile_note = {True: "The extracted code COMPILES (`jac check` passed).",
                    False: "The extracted code FAILS `jac check`.",
                    None: "No code block could be extracted from the response."}[compiles]
    return judged_llm(cache_key("code", item["id"], response), [
        {"role": "system", "content":
            "You grade a Jac-language assistant's coding response. " + rubric +
            " Weigh the compile-check result as strong evidence."},
        {"role": "user", "content":
            f"{task_desc}\n\n{compile_note}\n\nAssistant response:\n{response[:6000]}"},
    ], CodeJudge)


def judge_binary(kind: str, item: dict, turn_idx: int, response: str) -> dict | None:
    rubrics = {
        "small_talk": "The user made pure small talk. appropriate = the reply is a brief, "
                      "polite conversational response (it may mention being a Jac assistant) "
                      "with no fabricated technical content.",
        "off_topic": "The user asked about something unrelated to the Jac language. "
                     "appropriate = the reply politely declines or redirects toward Jac/Jaseci "
                     "topics instead of fully answering the off-topic request.",
        "multi_turn": "Given the conversation, appropriate = the reply is a coherent, on-point "
                      "continuation that addresses what the user actually asked for in their "
                      "last message.",
    }
    convo = "\n\n".join(f"User: {t['message'][:800]}" for t in item["turns"][:turn_idx + 1])
    return judged_llm(cache_key(f"bin-{kind}-{turn_idx}", item["id"], response), [
        {"role": "system", "content": "You audit one assistant reply. " + rubrics[kind]},
        {"role": "user", "content": f"{convo}\n\nAssistant reply:\n{response[:3000]}"},
    ], BinaryJudge)


def latest_runs() -> list[dict]:
    rows = {}
    for row in common.read_jsonl(common.RAW_RUNS_PATH):
        rows[(row["system"], row["item_id"], row["repeat"], row["turn"])] = row
    return list(rows.values())


def token_join() -> dict:
    """Aggregate proxy rows per turn attempt (uid keys can't double-count retries)."""
    agg: dict = {}
    for row in common.read_jsonl(common.PROXY_LOG_PATH):
        key = row.get("attempt") or (row.get("system"), row.get("item_id"),
                                     row.get("repeat"), row.get("turn"))
        slot = agg.setdefault(key, {"llm_calls": 0, "prompt_tokens": 0,
                                    "completion_tokens": 0, "total_tokens": 0,
                                    "cached_tokens": 0})
        slot["llm_calls"] += 1
        for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens"):
            slot[k] += row.get(k, 0) or 0
    return agg


def score_row(run: dict, item: dict, sections: dict[tuple, str],
              tokens: dict[tuple, dict]) -> dict:
    turn_idx = run["turn"]
    gold_agent = run["gold_agent"]
    response = run.get("response", "")
    out = dict(run)
    out["response"] = response[:300]  # keep judged.jsonl compact
    out["routing_correct"] = (run.get("routed_agent") == gold_agent)
    out["failed"] = bool(run.get("error")) or not response
    tok_key = run.get("attempt") or (run["system"], run["item_id"], run["repeat"], turn_idx)
    out.update(tokens.get(tok_key,
                          {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                           "total_tokens": 0, "cached_tokens": 0}))
    out["judge_score"] = None
    out["acceptable"] = None
    out["appropriate"] = None
    out["compiles"] = None
    if out["failed"]:
        return out

    category = item["category"]
    is_code_turn = gold_agent in CODE_AGENTS
    if is_code_turn:
        code = common.extract_jac_code(response)
        out["compiles"] = common.jac_check(code)[0] if code else None

    if category == "rag_qa":
        section = sections.get((item.get("source_file"), item.get("source_section")), "")
        v = judge_rag(item, response, section)
        if v:
            out["judge_score"], out["acceptable"] = v["score"], v["acceptable"]
    elif category in ("coding", "debugging"):
        v = judge_code(item, response, out["compiles"])
        if v:
            out["judge_score"] = v["score"]
            out["acceptable"] = v["score"] >= 4
    elif category in ("small_talk", "off_topic"):
        v = judge_binary(category, item, turn_idx, response)
        if v:
            out["appropriate"] = v["appropriate"]
    elif category == "multi_turn":
        if turn_idx == 0 and item.get("gold_answer") and gold_agent == "RagChat":
            section = sections.get((item.get("source_file"), item.get("source_section")), "")
            v = judge_rag(item, response, section)
            if v:
                out["judge_score"], out["acceptable"] = v["score"], v["acceptable"]
        else:
            v = judge_binary("multi_turn", item, turn_idx, response)
            if v:
                out["appropriate"] = v["appropriate"]
    return out


def main() -> None:
    items = {it["id"]: it for it in common.read_jsonl(common.DATASET_PATH)}
    runs = latest_runs()
    if not runs:
        raise SystemExit("no raw runs — run run_eval.py first")
    load_cache()
    sections = build_section_map()
    tokens = token_join()
    print(f"scoring {len(runs)} turns ({len(_cache)} cached judgments)")

    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(score_row, run, items[run["item_id"]], sections, tokens)
                   for run in runs if run["item_id"] in items]
        for i, fut in enumerate(futures, 1):
            results.append(fut.result())
            if i % 100 == 0:
                print(f"  {i}/{len(futures)}")

    common.JUDGED_PATH.unlink(missing_ok=True)
    for row in results:
        common.append_jsonl(common.JUDGED_PATH, row)
    print(f"judged {len(results)} turns -> {common.JUDGED_PATH}")


if __name__ == "__main__":
    main()
