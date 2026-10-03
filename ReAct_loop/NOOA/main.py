"""BFCL v4 web_search in NVIDIA NOOA: one agent, CodeAct over its tool methods.

The tools are ordinary methods; `solve` has an ellipsis body, so NOOA
implements it with CodeAct: the model writes Python that calls the tools and
ends with `return_result`. The step budget is CodeAct's `max_iterations`.

    python NOOA/main.py web_search_0
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Annotated, Any, Callable, Literal

import html2text
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from ddgs.base import BaseSearchEngine
from nooa import Agent, strategy
from nooa.config import CodeActConfig
from nooa.strategies import CodeActStrategy
from nooa.unifiedllm import CompletionClient, LLMResponse

MODEL = "glm-5.2"
MODEL_BASE = "https://ollama.com/v1"
API_KEY = os.environ.get("OLLAMA_API_KEY", "")
TEMPERATURE = 1.0
MAX_TOKENS = 4096
MAX_STEPS = 16
MAX_CONTENT_CHARS = 20000   # cap on fetch_url_content output; BFCL itself has none
SEARCH_RETRIES = 5          # retries of a search that got HTTP 429 (waits of 1, 2, 4, 8, 15 min) before the run is abandoned
MIN_SEARCH_INTERVAL_S = 10  # engines block bursts: live searches at least this far apart, across every run and arm
RATE_LIMIT_EXIT = 75        # exit status of an abandoned run; eval/run.py records it as an error, to rerun
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "BFCL_v4_web_search.json"
ANSWERS = ROOT / "data" / "BFCL_v4_web_search_answers.json"
CACHE_DIR = ROOT / "web_cache"
SEARCH_CLOCK = CACHE_DIR / ".last_search"   # time of the last live search request

# BFCL's allowed values for search_engine_query(region=...) and fetch_url_content(mode=...)
REGIONS = (
    "xa-ar", "xa-en", "ar-es", "au-en", "at-de", "be-fr", "be-nl", "br-pt", "bg-bg", "ca-en", "ca-fr", "ct-ca",
    "cl-es", "cn-zh", "co-es", "hr-hr", "cz-cs", "dk-da", "ee-et", "fi-fi", "fr-fr", "de-de", "gr-el", "hk-tz",
    "hu-hu", "in-en", "id-id", "id-en", "ie-en", "il-he", "it-it", "jp-jp", "kr-kr", "lv-lv", "lt-lt", "xl-es",
    "my-ms", "my-en", "mx-es", "nl-nl", "nz-en", "no-no", "pe-es", "ph-en", "ph-tl", "pl-pl", "pt-pt", "ro-ro",
    "ru-ru", "sg-en", "sk-sk", "sl-sl", "za-en", "es-es", "se-sv", "ch-de", "ch-fr", "ch-it", "tw-tz", "th-th",
    "tr-tr", "ua-uk", "uk-en", "us-en", "ue-es", "ve-es", "vn-vi", "wt-wt",
)
MODES = ("raw", "markdown", "truncate")

USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}   # filled by this arm's own hook


class CountingClient(CompletionClient):
    """NOOA's litellm client, counting calls and tokens."""

    async def acall(self, messages: list[dict[str, Any]], *args: Any, **kwargs: Any) -> LLMResponse:
        response = await super().acall(messages, *args, **kwargs)
        usage = response.usage or {}
        USAGE["calls"] += 1
        USAGE["prompt_tokens"] += usage.get("prompt_tokens") or 0
        USAGE["completion_tokens"] += usage.get("completion_tokens") or 0
        return response


llm = CountingClient(
    model=f"openai/{MODEL}", api_base=MODEL_BASE, api_key=API_KEY, temperature=TEMPERATURE, max_tokens=MAX_TOKENS
)


class Solver(Agent, llm=llm):
    def search_engine_query(
        self,
        keywords: Annotated[str, "The keywords to search for."],
        max_results: Annotated[int, "The maximum number of search results to return."] = 10,
        region: Annotated[Literal[REGIONS], "The region to search in. Defaults to 'wt-wt' (No region)."] = "wt-wt",
    ) -> str:
        """This tool belongs to the Web Search API category. It provides functions to search the web and browse search results. Tool description: This function queries the search engine for the provided keywords and region."""
        return call_web_tool("search_engine_query", {"keywords": keywords, "max_results": max_results, "region": region})

    def fetch_url_content(
        self,
        url: Annotated[str, "The URL to fetch content from. Must start with 'http://' or 'https://'."],
        mode: Annotated[
            Literal[MODES],
            "The mode to process the fetched content. `raw` returns the raw HTML content; `markdown` converts raw HTML "
            "content to Markdown format for better readability, using html2text; `truncate` extracts and cleans text "
            "by removing scripts, styles, and extraneous whitespace.",
        ] = "raw",
    ) -> str:
        """This tool belongs to the Web Search API category. It provides functions to search the web and browse search results. Tool description: This function retrieves content from the provided URL and processes it based on the selected mode."""
        return call_web_tool("fetch_url_content", {"url": url, "mode": mode})

    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=MAX_STEPS)))
    async def solve(self, question: Annotated[str, "The multi-hop question to answer."]) -> str:
        """Answer the question by searching and reading pages; each tool call should resolve one hop of the question. Finish as soon as the answer is established, with the answer alone, in the format the question asks for. Never repeat a tool call already made: if a result did not resolve the hop, change the query or open a different page. If three calls in a row add nothing new, finish with the best answer so far."""
        ...


def solve(question: str) -> str:
    return asyncio.run(Solver().solve(question))


# ---------------------------------------------------------------------------
# Task environment: identical in every Python arm (tools, data, grading).
# The two tools are BFCL's WebSearchAPI, with the search backend swapped from
# SerpAPI (DuckDuckGo) to ddgs' auto backend (rotates across engines) and fetch output capped.
# ---------------------------------------------------------------------------

CALLS: list[str] = []   # every tool call of this task, as "tool:arguments", in call order

FETCH_HEADERS = {   # BFCL's browser-like headers, minus "br" (no brotli decoder installed)
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Referer": "https://www.google.com/",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-User": "?1",
    "Sec-Fetch-Dest": "document",
}


def call_web_tool(name: str, args: dict[str, Any]) -> str:
    # one tool call, rendered to text the way BFCL's executor renders it, cached by tool name + arguments
    call = f"{name}:{json.dumps(args, sort_keys=True)}"
    CALLS.append(call)
    path = CACHE_DIR / (hashlib.md5(call.encode()).hexdigest() + ".txt")
    if path.exists():
        return path.read_text()
    try:
        result = WEB_TOOLS[name](**args)
    except Exception as e:
        return f"Error during execution: {e}"
    text = result if isinstance(result, str) else json.dumps(result) if isinstance(result, dict) else str(result)
    if not (isinstance(result, dict) and "error" in result):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return text


RATE_LIMITED: list[str] = []   # engines that answered HTTP 429 during the current search


def engine_request(engine: BaseSearchEngine, *args: Any, **kwargs: Any) -> str | None:
    # ddgs treats every non-200 page as "no results"; note the 429s so that a search left empty by
    # rate limits is retried instead of reaching the agent as "No results found."
    resp = engine.http_client.request(*args, **kwargs)
    if resp.status_code == 429:
        RATE_LIMITED.append(engine.name)
    return resp.text if resp.status_code == 200 else None


BaseSearchEngine.request = engine_request


def wait_for_search_slot() -> None:
    # engines block bursts of searches: keep live ones MIN_SEARCH_INTERVAL_S apart across every run and arm
    try:
        last = float(SEARCH_CLOCK.read_text())
    except (OSError, ValueError):
        last = 0.0
    time.sleep(max(0.0, last + MIN_SEARCH_INTERVAL_S - time.time()))
    SEARCH_CLOCK.parent.mkdir(parents=True, exist_ok=True)
    SEARCH_CLOCK.write_text(str(time.time()))


def bfcl_search_engine_query(keywords: str, max_results: int = 10, region: str = "wt-wt") -> list | dict:
    backoff = 60
    for attempt in range(SEARCH_RETRIES + 1):
        wait_for_search_slot()
        RATE_LIMITED.clear()
        try:
            hits = DDGS().text(keywords, region=region, max_results=max_results, backend="auto")
            break
        except Exception as e:
            if not RATE_LIMITED:
                return {"error": str(e)}
            if attempt == SEARCH_RETRIES:
                # the search backend is down, not the agent: end the process (every framework would hand an
                # exception back to the model) so eval/run.py records an error and reruns the task
                engines = ", ".join(sorted(set(RATE_LIMITED)))
                print(f"search rate-limited: HTTP 429 from {engines} after {SEARCH_RETRIES} retries (query: {keywords})",
                      file=sys.stderr, flush=True)
                os._exit(RATE_LIMIT_EXIT)
            time.sleep(backoff + random.uniform(0, 10))
            backoff = min(backoff * 2, 900)
    if not hits:
        return {"error": "Failed to retrieve the search results from server. Please try again later."}
    return [{"title": h["title"], "href": h["href"], "body": h["body"]} for h in hits[:max_results]]


def bfcl_fetch_url_content(url: str, mode: str = "raw") -> dict:
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"Invalid URL: {url}")
    try:
        response = requests.get(url, headers=FETCH_HEADERS, timeout=20, allow_redirects=True)
        response.raise_for_status()
        if mode == "raw":
            content = response.text
        elif mode == "markdown":
            content = html2text.HTML2Text().handle(response.text)
        elif mode == "truncate":
            soup = BeautifulSoup(response.text, "html.parser")
            for script_or_style in soup(["script", "style"]):
                script_or_style.extract()
            content = soup.get_text(separator="\n", strip=True)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        return {"content": content[:MAX_CONTENT_CHARS]}
    except Exception as e:
        return {"error": f"An error occurred while fetching {url}: {str(e)}"}


WEB_TOOLS: dict[str, Callable[..., Any]] = {
    "search_engine_query": bfcl_search_engine_query,
    "fetch_url_content": bfcl_fetch_url_content,
}


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def matches(answer: str, gold: list[str]) -> bool:
    a = normalize(answer)
    return any(n and (n == a or n in a) for n in map(normalize, gold))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_question(task_id: str) -> str:
    for row in load_jsonl(DATA):
        if row["id"] == task_id:
            return str(row["question"][0][0]["content"])
    raise KeyError(f"{task_id} not in {DATA}")


def load_gold(task_id: str) -> list[str]:
    if ANSWERS.exists():
        for row in load_jsonl(ANSWERS):
            if row["id"] == task_id:
                return [str(g) for g in row["ground_truth"]]
    return []


def run(solve: Callable[[str], str]) -> None:
    task_id = sys.argv[1] if len(sys.argv) > 1 else "web_search_0"
    question = load_question(task_id)
    gold = load_gold(task_id)
    errors: list[str] = []
    try:
        answer = solve(question)
    except Exception as e:
        errors.append(f"solve failed: {e}")
        answer = ""
    verdict = "UNGRADED" if not gold else ("PASSED" if matches(answer, gold) else "FAILED")
    print(f"Verdict: {verdict}")
    print(f"Answer: {answer}")
    print(f"Gold: {' | '.join(gold)}")
    print(f"Tool calls: {len(CALLS)}")
    print(f"Usage: calls={USAGE['calls']} prompt_tokens={USAGE['prompt_tokens']} completion_tokens={USAGE['completion_tokens']}")
    if errors:
        print("Diagnostics:")
        for error in errors:
            print(f"- {error}")


if __name__ == "__main__":
    run(solve)
