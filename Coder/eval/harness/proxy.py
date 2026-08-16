"""Token-counting reverse proxy in front of the OpenAI API.

All three systems under test are pointed at this proxy via
OPENAI_BASE_URL=http://127.0.0.1:8899/v1, so every LLM call they make —
router, ReAct iterations, retries — passes through here and is logged with
its token usage, attributed to whatever marker the eval runner last set via
POST /__mark. This gives identical, neutral accounting across byllm/litellm
and langchain-openai without modifying the systems.

Run:  /home/xiaoyu/miniconda3/envs/jaseci/bin/python proxy.py
Env:  PROXY_UPSTREAM (default https://api.openai.com), PROXY_PORT, PROXY_LOG
"""

import json
import os
import sys
import time
from pathlib import Path

from aiohttp import ClientSession, ClientTimeout, web

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402

UPSTREAM = os.environ.get("PROXY_UPSTREAM", "https://api.openai.com").rstrip("/")
PORT = int(os.environ.get("PROXY_PORT", str(common.PROXY_PORT)))
LOG_PATH = Path(os.environ.get("PROXY_LOG", str(common.PROXY_LOG_PATH)))

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
    # aiohttp hands us the DECOMPRESSED body, so the upstream's encoding header
    # must not be forwarded — clients would try to gunzip plain JSON and fail.
    "content-encoding",
}

# Attribution marker set by the runner; requests are logged under the current one.
marker: dict = {"system": "", "item_id": "", "repeat": 0, "turn": 0, "attempt": ""}


def extract_usage(payload: dict) -> dict | None:
    """Normalize usage from chat-completions ({prompt,completion}_tokens) or
    responses-API ({input,output}_tokens) payloads."""
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion = usage.get("completion_tokens", usage.get("output_tokens"))
    if prompt is None and completion is None:
        return None
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    return {
        "prompt_tokens": int(prompt or 0),
        "completion_tokens": int(completion or 0),
        "total_tokens": int(usage.get("total_tokens") or (prompt or 0) + (completion or 0)),
        "cached_tokens": int(details.get("cached_tokens") or 0),
    }


def log_row(path: str, status: int, model: str, usage: dict | None) -> None:
    row = {
        "ts": time.time(),
        **marker,
        "path": path,
        "status": status,
        "model": model,
        **(usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cached_tokens": 0}),
        "has_usage": usage is not None,
    }
    common.append_jsonl(LOG_PATH, row)


async def handle_mark(request: web.Request) -> web.Response:
    body = await request.json()
    marker.update({k: body.get(k, marker.get(k))
                   for k in ("system", "item_id", "repeat", "turn", "attempt")})
    return web.json_response({"ok": True, "marker": marker})


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({"ok": True, "upstream": UPSTREAM, "log": str(LOG_PATH)})


async def handle_proxy(request: web.Request) -> web.StreamResponse:
    upstream_url = f"{UPSTREAM}{request.rel_url}"
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
    body = await request.read()

    session: ClientSession = request.app["client"]
    async with session.request(request.method, upstream_url, headers=headers, data=body) as resp:
        content_type = resp.headers.get("Content-Type", "")
        resp_headers = {k: v for k, v in resp.headers.items() if k.lower() not in HOP_BY_HOP}

        if "text/event-stream" in content_type:
            # Stream through while scanning SSE data lines for a usage payload.
            out = web.StreamResponse(status=resp.status, headers=resp_headers)
            await out.prepare(request)
            usage = None
            model = ""
            buffer = b""
            async for chunk in resp.content.iter_any():
                await out.write(chunk)
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        continue
                    try:
                        payload = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    model = payload.get("model", model)
                    usage = extract_usage(payload) or usage
            await out.write_eof()
            log_row(str(request.rel_url.path), resp.status, model, usage)
            return out

        raw = await resp.read()
        usage = None
        model = ""
        try:
            payload = json.loads(raw)
            model = payload.get("model", "")
            usage = extract_usage(payload)
        except (json.JSONDecodeError, AttributeError):
            pass
        log_row(str(request.rel_url.path), resp.status, model, usage)
        return web.Response(status=resp.status, headers=resp_headers, body=raw)


async def make_app() -> web.Application:
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app["client"] = ClientSession(timeout=ClientTimeout(total=600))
    app.router.add_post("/__mark", handle_mark)
    app.router.add_get("/__health", handle_health)
    app.router.add_route("*", "/{tail:.*}", handle_proxy)

    async def close_client(app):
        await app["client"].close()

    app.on_cleanup.append(close_client)
    return app


if __name__ == "__main__":
    common.setup_env()
    print(f"proxy: 127.0.0.1:{PORT} -> {UPSTREAM}   log: {LOG_PATH}")
    web.run_app(make_app(), host="127.0.0.1", port=PORT, print=None)
