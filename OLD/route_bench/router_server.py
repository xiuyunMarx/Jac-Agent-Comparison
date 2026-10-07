#!/usr/bin/env python3
"""Tiny HTTP front for an LLMRouter router, spoken to by the byLLM runtime.

byLLM (with `[byllm.routing]` enabled) POSTs
    {"query": <request text with the call-site header>, "candidates": [...],
     "site": <function qualname>, "base_model": <configured model>}
and expects {"model_name": <one of candidates>}.

Policies:
  knn       LLMRouter KNNRouter trained by build_router.py (default)
  largest   always the last candidate  (LLMRouter's largest_llm baseline)
  smallest  always the first candidate (LLMRouter's smallest_llm baseline)

Run inside the llmrouter venv (~/.venvs/llmrouter), CPU only:
    LLMROUTER_EMBEDDING_DEVICE=cpu python router_server.py --config out/knnrouter_test.yaml
"""
import argparse
import json
import os
import sys
import threading
import time
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

warnings.filterwarnings("ignore")
os.environ.setdefault("LLMROUTER_EMBEDDING_DEVICE", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


class Policy:
    def __init__(self, name, config):
        self.name = name
        self.router = None
        self.lock = threading.Lock()
        if name == "knn":
            from llmrouter.models.knnrouter import KNNRouter
            self.router = KNNRouter(yaml_path=config)
            # warm the embedding model so the first agent call is not slow
            self.router.route_single({"query": "warm-up"})

    def choose(self, query, candidates, base_model):
        if self.name == "largest":
            return candidates[-1], "largest"
        if self.name == "smallest":
            return candidates[0], "smallest"
        with self.lock:
            out = self.router.route_single({"query": query})
        name = out.get("model_name", "")
        if name not in candidates:
            return base_model or candidates[-1], f"knn_unknown:{name}"
        return name, "knn"


def make_handler(policy, log_path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
                cands = list(req.get("candidates") or [])
                t0 = time.time()
                model, why = policy.choose(req.get("query", ""), cands, req.get("base_model"))
                body = {"model_name": model, "policy": why}
                if log_path:
                    with open(log_path, "a") as f:
                        f.write(json.dumps({"ts": t0, "site": req.get("site"), "model": model,
                                            "policy": why, "ms": round((time.time() - t0) * 1000, 1)}) + "\n")
                code = 200
            except Exception as exc:  # never take the agent down
                body, code = {"error": str(exc)}, 500
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            data = json.dumps({"ok": True, "policy": policy.name}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="knn", choices=["knn", "largest", "smallest"])
    ap.add_argument("--config", help="LLMRouter yaml (test config with load_model_path) for --policy knn")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--log", default="", help="append one JSON line per routing decision")
    args = ap.parse_args()
    if args.policy == "knn" and not args.config:
        sys.exit("--config is required for the knn policy")
    policy = Policy(args.policy, args.config)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(policy, args.log))
    print(f"[router_server] {args.policy} ready on http://127.0.0.1:{args.port}/route", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
