#!/usr/bin/env python3
"""Serve the local ollama with thinking switched off.

Qwen3.6 thinks by default, and ollama has no server- or Modelfile-level switch
for it; the only off switch is `reasoning_effort: "none"` in each request. Not
every framework can send that (litellm drops it for a model it does not know,
CrewAI only sends it to models it recognises as reasoning models), so the
switch lives here, in the endpoint, where every arm gets exactly the same
model. Every chat-completion request gets `reasoning_effort: "none"` unless it
already carries the key; everything else is relayed as-is.

    ./local_nothink.py                 # 127.0.0.1:11436 -> 127.0.0.1:11435
    LISTEN=11436 UPSTREAM=http://127.0.0.1:11435 ./local_nothink.py
"""
import json
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.environ.get("UPSTREAM", "http://127.0.0.1:11435")
LISTEN = int(os.environ.get("LISTEN", "11436"))
HOP = {"connection", "transfer-encoding", "content-length", "keep-alive"}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _relay(self, body):
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP | {"host"}}
        req = urllib.request.Request(UPSTREAM + self.path, body, headers, method=self.command)
        try:
            resp = urllib.request.urlopen(req, timeout=3600)
        except urllib.error.HTTPError as err:
            resp = err
        self.send_response(int(resp.status or 502))
        for k, v in resp.headers.items():
            if k.lower() not in HOP:
                self.send_header(k, v)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        while True:                       # stream through, so SSE clients see chunks as they come
            chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
            if not chunk:
                break
            self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
            self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.path.endswith("/chat/completions"):
            try:
                payload = json.loads(body)
            except ValueError:
                payload = None
            if isinstance(payload, dict):
                payload.setdefault("reasoning_effort", "none")
                body = json.dumps(payload).encode()
        self._relay(body)

    def do_GET(self):
        self._relay(None)

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    print(f"local_nothink: {LISTEN} -> {UPSTREAM}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", LISTEN), Handler).serve_forever()
