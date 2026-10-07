#!/usr/bin/env python3
"""Turn byLLM prompt logs into LLMRouter training data and train a KNN router.

Input: BYLLM_PROMPT_LOG files from a run of the agent on its configured
(strong) model. Each line is one request exactly as the runtime sent it
(messages, tools, response_format) plus `route_query`, the text the runtime
hands to the router (call-site header + messages).

For every logged request, every candidate model answers the *same* request
(replay). The runtime's own contract scores the answer with no human labels:

  valid  = the reply satisfies the declared contract -- parses against the
           response_format JSON schema for typed sites, or is a well-formed
           call to a declared tool for tool sites
  agree  = the reply matches the reference model's reply on the typed skeleton
           (enum/bool/number/short-string fields, or the tool called)
  perf   = valid * agree
  performance (the KNN label) = perf - lambda * cost / max_cost(query)

so among candidates that satisfy the contract the cheaper one wins. Outputs
LLMRouter's standard files + train/test YAMLs, trains KNNRouter, and reports
held-out routing accuracy against the per-query oracle.

Run inside ~/.venvs/llmrouter (CPU only):
  build_router.py --prompts run/prompts.jsonl --out out \
      --candidates openai/gemma4:31b openai/glm-5.2 --reference openai/glm-5.2
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import random
import re
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("LLMROUTER_EMBEDDING_DEVICE", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

HERE = Path(__file__).resolve().parent
PRICES = json.loads((HERE / "pricing.json").read_text())   # model -> [in, cached_in, out] USD per 1M


def bare(model):
    return model.split("/", 1)[1] if "/" in model else model


def price(model, prompt_tokens, completion_tokens):
    p = PRICES.get(bare(model)) or PRICES.get(model)
    if not p:
        raise SystemExit(f"no price for {model} in pricing.json")
    return (prompt_tokens * p[0] + completion_tokens * p[2]) / 1e6


# ---------------------------------------------------------------- replay

def load_requests(paths):
    reqs, seen = [], set()
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            if not r.get("messages") or not r.get("route_query"):
                continue
            key = hashlib.sha1(json.dumps([r["messages"], r.get("tools"), r.get("response_format")],
                                          sort_keys=True, default=str).encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            reqs.append({"id": key[:12], "scope": r["scope"], "messages": r["messages"],
                         "tools": r.get("tools"), "response_format": r.get("response_format"),
                         "route_query": r["route_query"]})
    return reqs


def call(client, model, req, temperature):
    kw = dict(model=bare(model), messages=req["messages"], temperature=temperature, max_tokens=4096)
    if req["tools"]:
        kw["tools"] = req["tools"]
    if req["response_format"]:
        kw["response_format"] = req["response_format"]
    last = None
    for attempt in range(3):
        try:
            t0 = time.time()
            resp = client.chat.completions.create(**kw)
            msg = resp.choices[0].message
            return {
                "content": msg.content or "",
                "tool_calls": [{"name": tc.function.name, "arguments": tc.function.arguments}
                               for tc in (msg.tool_calls or [])],
                "prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens,
                "latency_ms": round((time.time() - t0) * 1000), "error": None,
            }
        except Exception as exc:  # rate limit / transient
            last = str(exc)
            time.sleep(3 * (attempt + 1))
    return {"content": "", "tool_calls": [], "prompt_tokens": 0, "completion_tokens": 0,
            "latency_ms": 0, "error": last}


# ---------------------------------------------------------------- contract scoring

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text):
    text = text.strip()
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    a, b = text.find("{"), text.rfind("}")
    if a != -1 and b > a:
        try:
            return json.loads(text[a:b + 1])
        except Exception:
            return None
    return None


def typed_value(req, ans):
    """Parsed typed output (unwrapped like the runtime does), or None if the contract fails."""
    import jsonschema
    schema = req["response_format"]["json_schema"]["schema"]
    obj = extract_json(ans["content"])
    if obj is None:
        return None
    # byLLM wraps non-object return types in {"schema_object_wrapper": ...} but accepts the bare value
    props = schema.get("properties") or {}
    if list(props) == ["schema_object_wrapper"] and not (isinstance(obj, dict) and "schema_object_wrapper" in obj):
        obj = {"schema_object_wrapper": obj}
    try:
        jsonschema.validate(obj, schema)
    except Exception:
        return None
    if isinstance(obj, dict) and list(obj) == ["schema_object_wrapper"]:
        return obj["schema_object_wrapper"]
    return obj


def skeleton(v):
    """The comparable part of a typed value: enums, numbers, bools, short strings."""
    if isinstance(v, dict):
        return {k: skeleton(x) for k, x in v.items() if skeleton(x) is not None}
    if isinstance(v, list):
        return [skeleton(x) for x in v] if all(skeleton(x) is not None for x in v) else len(v)
    if isinstance(v, str):
        s = v.strip()
        return s.lower() if (len(s) <= 48 and "\n" not in s) else None
    return v


def score(req, ans, ref):
    """(valid, agree) for one candidate answer against the reference answer."""
    if ans["error"]:
        return 0, 0
    if req["response_format"]:
        val = typed_value(req, ans)
        if val is None:
            return 0, 0
        rv = typed_value(req, ref) if ref else None
        if rv is None:
            return 1, 1
        return 1, int(skeleton(val) == skeleton(rv))
    if req["tools"]:
        declared = {t["function"]["name"] for t in req["tools"]}
        names = [tc["name"] for tc in ans["tool_calls"]]
        if not names or any(n not in declared for n in names):
            return 0, 0
        for tc in ans["tool_calls"]:
            try:
                json.loads(tc["arguments"] or "{}")
            except Exception:
                return 0, 0
        ref_names = [tc["name"] for tc in ref["tool_calls"]] if ref else []
        if not ref_names:
            return 1, 1
        return 1, int(names[0] == ref_names[0])
    return int(bool(ans["content"].strip())), 1


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", nargs="+", required=True, help="BYLLM_PROMPT_LOG files from the strong-model run")
    ap.add_argument("--out", required=True)
    ap.add_argument("--candidates", nargs="+", required=True, help="litellm model names, cheapest first")
    ap.add_argument("--reference", required=True, help="candidate whose answers define agreement")
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://ollama.com/v1"))
    ap.add_argument("--api-key", default=os.environ.get("OLLAMA_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    ap.add_argument("--lambda", dest="lam", type=float, default=0.2, help="cost weight in the KNN label")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--n-neighbors", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-replay", action="store_true", help="reuse out/replay.jsonl")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    assert args.reference in args.candidates

    reqs = load_requests(args.prompts)
    print(f"[build] {len(reqs)} distinct requests from {len(args.prompts)} log(s)", flush=True)

    # 1. replay every request on every candidate
    replay_path = out / "replay.jsonl"
    answers = {}
    if args.skip_replay and replay_path.exists():
        for line in open(replay_path):
            r = json.loads(line)
            answers[(r["id"], r["model"])] = r["answer"]
    else:
        from openai import OpenAI
        if not args.api_key:
            sys.exit("no API key (OLLAMA_API_KEY)")
        client = OpenAI(base_url=args.base_url, api_key=args.api_key)
        jobs = [(req, m) for req in reqs for m in args.candidates]
        with replay_path.open("w") as fh, cf.ThreadPoolExecutor(args.workers) as pool:
            futs = {pool.submit(call, client, m, req, args.temperature): (req, m) for req, m in jobs}
            for i, fut in enumerate(cf.as_completed(futs), 1):
                req, m = futs[fut]
                ans = fut.result()
                answers[(req["id"], m)] = ans
                fh.write(json.dumps({"id": req["id"], "scope": req["scope"], "model": m, "answer": ans}) + "\n")
                if i % 20 == 0 or i == len(jobs):
                    print(f"[replay] {i}/{len(jobs)}", flush=True)

    # 2. contract scoring + cost-adjusted label
    rows, per_site = [], {}
    for req in reqs:
        ref = answers.get((req["id"], args.reference))
        costs = {m: price(m, answers[(req["id"], m)]["prompt_tokens"], answers[(req["id"], m)]["completion_tokens"])
                 for m in args.candidates}
        cmax = max(costs.values()) or 1.0
        for m in args.candidates:
            ans = answers[(req["id"], m)]
            valid, agree = score(req, ans, ref)
            perf = valid * agree
            rows.append({"id": req["id"], "task_name": req["scope"], "query": req["route_query"],
                         "ground_truth": "", "model_name": m, "response": (ans["content"] or json.dumps(ans["tool_calls"]))[:2000],
                         "valid": valid, "agree": agree, "task_performance": perf, "cost": costs[m],
                         "performance": perf - args.lam * costs[m] / cmax,
                         "token_num": ans["prompt_tokens"] + ans["completion_tokens"]})
            s = per_site.setdefault(req["scope"], {}).setdefault(m, {"n": 0, "valid": 0, "agree": 0, "cost": 0.0})
            s["n"] += 1; s["valid"] += valid; s["agree"] += perf; s["cost"] += costs[m]
    print("[label] per site / model: valid rate, contract-agreement rate, mean cost")
    for site, ms in per_site.items():
        for m, s in ms.items():
            print(f"  {site.rsplit('.',1)[-1]:24s} {m:22s} valid={s['valid']/s['n']:.2f} agree={s['agree']/s['n']:.2f} cost=${s['cost']/s['n']:.5f} n={s['n']}")

    # 3. split by request, embed, write LLMRouter files
    ids = sorted({r["id"] for r in rows})
    random.Random(args.seed).shuffle(ids)
    n_test = max(1, int(len(ids) * args.test_frac))
    test_ids = set(ids[:n_test])
    queries = {}
    for req in reqs:
        queries.setdefault(req["route_query"], len(queries))
    from llmrouter.utils import get_longformer_embedding
    import torch
    texts = list(queries)
    embs = []
    for i in range(0, len(texts), 8):
        e = get_longformer_embedding(texts[i:i + 8])
        embs.append(e if e.dim() == 2 else e.unsqueeze(0))
        print(f"[embed] {min(i+8, len(texts))}/{len(texts)}", flush=True)
    emb = torch.cat(embs, 0).cpu()
    torch.save(emb, out / "query_embeddings_longformer.pt")

    def dump(path, items):
        with open(path, "w") as fh:
            for it in items:
                fh.write(json.dumps(it) + "\n")
    tr = [dict(r, embedding_id=queries[r["query"]]) for r in rows if r["id"] not in test_ids]
    te = [dict(r, embedding_id=queries[r["query"]]) for r in rows if r["id"] in test_ids]
    dump(out / "routing_train.jsonl", tr)
    dump(out / "routing_test.jsonl", te)
    qrows = {}
    for req in reqs:
        qrows[req["id"]] = {"task_name": req["scope"], "query": req["route_query"], "ground_truth": "",
                            "metric": "contract", "choices": None, "task_id": req["id"]}
    dump(out / "query_train.jsonl", [q for i, q in qrows.items() if i not in test_ids])
    dump(out / "query_test.jsonl", [q for i, q in qrows.items() if i in test_ids])
    llm_data = {m: {"size": "", "feature": f"{m} via {args.base_url}", "input_price": PRICES[bare(m)][0],
                    "output_price": PRICES[bare(m)][2], "model": bare(m), "service": "ollama",
                    "api_endpoint": args.base_url} for m in args.candidates}
    (out / "llm_candidates.json").write_text(json.dumps(llm_data, indent=2))
    (out / "llm_embeddings.json").write_text(json.dumps({m: dict(v, embedding=[]) for m, v in llm_data.items()}, indent=2))

    import yaml
    data_path = {"query_data_train": str(out / "query_train.jsonl"), "query_data_test": str(out / "query_test.jsonl"),
                 "query_embedding_data": str(out / "query_embeddings_longformer.pt"),
                 "routing_data_train": str(out / "routing_train.jsonl"), "routing_data_test": str(out / "routing_test.jsonl"),
                 "llm_data": str(out / "llm_candidates.json"), "llm_embedding_data": str(out / "llm_embeddings.json")}
    n_train_queries = len({r["query"] for r in tr})
    k = max(1, min(args.n_neighbors, n_train_queries))     # tiny arms: fewer training queries than k
    hparam = {"n_neighbors": k, "weights": "distance", "algorithm": "brute", "metric": "cosine", "n_jobs": 1}
    pkl = str(out / "knnrouter.pkl")
    (out / "knnrouter_train.yaml").write_text(yaml.safe_dump({
        "data_path": data_path, "model_path": {"ini_model_path": "", "save_model_path": pkl},
        "metric": {"weights": {"performance": 1, "cost": 0, "llm_judge": 0}}, "hparam": hparam}))
    (out / "knnrouter_test.yaml").write_text(yaml.safe_dump({
        "data_path": data_path, "model_path": {"load_model_path": pkl},
        "metric": {"weights": {"performance": 1, "cost": 0, "llm_judge": 0}}, "hparam": hparam}))

    # 4. train (LLMRouter's own classes) and evaluate on the held-out requests
    from llmrouter.models.knnrouter import KNNRouter, KNNRouterTrainer
    router = KNNRouter(yaml_path=str(out / "knnrouter_train.yaml"))
    KNNRouterTrainer(router).train()
    router = KNNRouter(yaml_path=str(out / "knnrouter_test.yaml"))
    by_q = {}
    for r in te:
        by_q.setdefault(r["query"], {})[r["model_name"]] = r
    hits, n, cost_routed, cost_strong, cost_cheap, perf_routed, perf_strong, perf_cheap = 0, 0, 0, 0, 0, 0, 0, 0
    picks = {}
    for q, ms in by_q.items():
        oracle = max(ms.values(), key=lambda r: r["performance"])["model_name"]
        pick = router.route_single({"query": q})["model_name"]
        picks[pick] = picks.get(pick, 0) + 1
        n += 1; hits += int(pick == oracle)
        strong, cheap = ms[args.candidates[-1]], ms[args.candidates[0]]
        cost_routed += ms[pick]["cost"]; perf_routed += ms[pick]["task_performance"]
        cost_strong += strong["cost"]; perf_strong += strong["task_performance"]
        cost_cheap += cheap["cost"]; perf_cheap += cheap["task_performance"]
    summary = {"requests": len(reqs), "train_requests": len(reqs) - n_test, "test_requests": n,
               "router_oracle_accuracy": round(hits / max(n, 1), 3), "test_picks": picks,
               "offline_test": {"routed": {"contract_perf": round(perf_routed / max(n, 1), 3), "cost_usd": round(cost_routed, 5)},
                                "strong": {"contract_perf": round(perf_strong / max(n, 1), 3), "cost_usd": round(cost_strong, 5)},
                                "cheap": {"contract_perf": round(perf_cheap / max(n, 1), 3), "cost_usd": round(cost_cheap, 5)}},
               "lambda": args.lam, "k": k, "candidates": args.candidates, "reference": args.reference}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
