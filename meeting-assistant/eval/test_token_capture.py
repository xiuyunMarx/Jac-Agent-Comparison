"""Offline test for the CrewAI-side token capture (mock_outputs.py).

Boots a local fake OpenAI endpoint and drives the real openai SDK through the
patched transport, so the counting is verified end-to-end without an API key:

    /home/xiaoyu/miniconda3/envs/jaseci/bin/python test_token_capture.py

Run with the Python environment that has the openai SDK installed (the same
one that runs the CrewAI implementation).
"""

import asyncio
import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "CrewAI" / "src"))

from meeting_assistant_flow.utils import mock_outputs  # noqa: E402

CHAT_COMPLETION = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
}

EMBEDDING = {
    "object": "list",
    "data": [{"object": "embedding", "index": 0, "embedding": [0.0]}],
    "model": "text-embedding-3-small",
    "usage": {"prompt_tokens": 2, "total_tokens": 2},
}


class FakeOpenAI(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        body = CHAT_COMPLETION if "/chat/completions" in self.path else EMBEDDING
        payload = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


class TokenCaptureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenAI)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}/v1"
        mock_outputs.register_token_tracking()
        mock_outputs.register_token_tracking()  # idempotent: must not double-wrap

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def snapshot(self):
        return mock_outputs.token_usage.snapshot()

    def test_sync_chat_completion_is_counted_once(self):
        import openai

        before = self.snapshot()
        client = openai.OpenAI(api_key="test-key", base_url=self.base_url)
        client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}]
        )
        after = self.snapshot()
        self.assertEqual(after["calls"], before["calls"] + 1)
        self.assertEqual(after["prompt_tokens"], before["prompt_tokens"] + 11)
        self.assertEqual(after["completion_tokens"], before["completion_tokens"] + 7)

    def test_async_chat_completion_is_counted(self):
        import openai

        before = self.snapshot()

        async def call():
            client = openai.AsyncOpenAI(api_key="test-key", base_url=self.base_url)
            await client.chat.completions.create(
                model="gpt-4o", messages=[{"role": "user", "content": "hi"}]
            )
            await client.close()

        asyncio.run(call())
        after = self.snapshot()
        self.assertEqual(after["calls"], before["calls"] + 1)
        self.assertEqual(after["prompt_tokens"], before["prompt_tokens"] + 11)

    def test_non_llm_endpoints_are_ignored(self):
        import openai

        before = self.snapshot()
        client = openai.OpenAI(api_key="test-key", base_url=self.base_url)
        client.embeddings.create(model="text-embedding-3-small", input="hi")
        self.assertEqual(self.snapshot(), before)

    def test_responses_api_key_style(self):
        class Usage:
            input_tokens = 5
            output_tokens = 3

        class Resp:
            usage = Usage()

        before = self.snapshot()
        mock_outputs.token_usage.track("/responses", Resp())
        after = self.snapshot()
        self.assertEqual(after["calls"], before["calls"] + 1)
        self.assertEqual(after["prompt_tokens"], before["prompt_tokens"] + 5)
        self.assertEqual(after["completion_tokens"], before["completion_tokens"] + 3)

    def test_stream_without_usage_counts_call_only(self):
        before = self.snapshot()
        mock_outputs.token_usage.track("/chat/completions", object())
        after = self.snapshot()
        self.assertEqual(after["calls"], before["calls"] + 1)
        self.assertEqual(after["prompt_tokens"], before["prompt_tokens"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
