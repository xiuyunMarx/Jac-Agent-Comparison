import os;
import re;
import sys;
import json;
import time;
import random;
import hashlib;
import requests;
import html2text;
import from bs4 { BeautifulSoup, Tag }
import from ddgs { DDGS }
import from ddgs.base { BaseSearchEngine }
import from jaclang.byllm.lib { Model }

glob llm = Model(
    model_name="openai/glm-5.2",
    api_key=os.environ.get("OLLAMA_API_KEY", ""),
    config={"base_url": "https://ollama.com/v1"}
);

glob HERE: str = os.path.dirname(os.path.abspath(__file__));
glob DATA: str = os.path.join(HERE, "..", "data", "BFCL_v4_web_search.json");
glob ANSWERS: str = os.path.join(HERE, "..", "data", "BFCL_v4_web_search_answers.json");
glob CACHE_DIR: str = os.path.join(HERE, "..", "web_cache");
glob MAX_STEPS: int = 16;
glob MAX_CONTENT_CHARS: int = 20000;   # cap on fetch_url_content output; BFCL itself has none
glob SEARCH_RETRIES: int = 5;          # retries of a search that got HTTP 429 (waits of 1, 2, 4, 8, 15 min) before the run is abandoned
glob MIN_SEARCH_INTERVAL_S: int = 10;  # engines block bursts: live searches at least this far apart, across every run and arm
glob SEARCH_CLOCK: str = os.path.join(CACHE_DIR, ".last_search");   # time of the last live search request
glob RATE_LIMIT_EXIT: int = 75;        # exit status of an abandoned run; eval/run.py records it as an error, to rerun

# BFCL's allowed values for search_engine_query(region=...) and fetch_url_content(mode=...)
enum Region: str {
    XA_AR = "xa-ar", XA_EN = "xa-en", AR_ES = "ar-es", AU_EN = "au-en", AT_DE = "at-de", BE_FR = "be-fr",
    BE_NL = "be-nl", BR_PT = "br-pt", BG_BG = "bg-bg", CA_EN = "ca-en", CA_FR = "ca-fr", CT_CA = "ct-ca",
    CL_ES = "cl-es", CN_ZH = "cn-zh", CO_ES = "co-es", HR_HR = "hr-hr", CZ_CS = "cz-cs", DK_DA = "dk-da",
    EE_ET = "ee-et", FI_FI = "fi-fi", FR_FR = "fr-fr", DE_DE = "de-de", GR_EL = "gr-el", HK_TZ = "hk-tz",
    HU_HU = "hu-hu", IN_EN = "in-en", ID_ID = "id-id", ID_EN = "id-en", IE_EN = "ie-en", IL_HE = "il-he",
    IT_IT = "it-it", JP_JP = "jp-jp", KR_KR = "kr-kr", LV_LV = "lv-lv", LT_LT = "lt-lt", XL_ES = "xl-es",
    MY_MS = "my-ms", MY_EN = "my-en", MX_ES = "mx-es", NL_NL = "nl-nl", NZ_EN = "nz-en", NO_NO = "no-no",
    PE_ES = "pe-es", PH_EN = "ph-en", PH_TL = "ph-tl", PL_PL = "pl-pl", PT_PT = "pt-pt", RO_RO = "ro-ro",
    RU_RU = "ru-ru", SG_EN = "sg-en", SK_SK = "sk-sk", SL_SL = "sl-sl", ZA_EN = "za-en", ES_ES = "es-es",
    SE_SV = "se-sv", CH_DE = "ch-de", CH_FR = "ch-fr", CH_IT = "ch-it", TW_TZ = "tw-tz", TH_TH = "th-th",
    TR_TR = "tr-tr", UA_UK = "ua-uk", UK_EN = "uk-en", US_EN = "us-en", UE_ES = "ue-es", VE_ES = "ve-es",
    VN_VI = "vn-vi", WT_WT = "wt-wt"
}

enum Mode: str { RAW = "raw", MARKDOWN = "markdown", TRUNCATE = "truncate" }

# every tool call of this task, as "tool:arguments", in call order
glob CALLS: list[str] = [];


def search_engine_query(keywords: str, max_results: int = 10, region: Region = Region.WT_WT) -> str {
    return call_web_tool("search_engine_query", {"keywords": keywords, "max_results": max_results, "region": str(region)});
}

sem search_engine_query = "This tool belongs to the Web Search API category. It provides functions to search the web and browse search results. Tool description: This function queries the search engine for the provided keywords and region.";
sem search_engine_query.keywords = "The keywords to search for.";
sem search_engine_query.max_results = "The maximum number of search results to return.";
sem search_engine_query.region = "The region to search in. Defaults to 'wt-wt' (No region).";


def fetch_url_content(url: str, mode: Mode = Mode.RAW) -> str {
    return call_web_tool("fetch_url_content", {"url": url, "mode": str(mode)});
}

sem fetch_url_content = "This tool belongs to the Web Search API category. It provides functions to search the web and browse search results. Tool description: This function retrieves content from the provided URL and processes it based on the selected mode.";
sem fetch_url_content.url = "The URL to fetch content from. Must start with 'http://' or 'https://'.";
sem fetch_url_content.mode = "The mode to process the fetched content. `raw` returns the raw HTML content; `markdown` converts raw HTML content to Markdown format for better readability, using html2text; `truncate` extracts and cleans text by removing scripts, styles, and extraneous whitespace.";


def solve(question: str) -> str by llm(
    tools=[search_engine_query, fetch_url_content],
    temperature=1.0,
    max_tokens=4096,
    max_react_iterations=MAX_STEPS
);

sem solve = "Answer the question by searching and reading pages; each tool call should resolve one hop of the question. Finish as soon as the answer is established, with the answer alone, in the format the question asks for. Never repeat a tool call already made: if a result did not resolve the hop, change the query or open a different page. If three calls in a row add nothing new, finish with the best answer so far.";
sem solve.question = "The multi-hop question to answer.";


# The two tools are BFCL's WebSearchAPI, with the search backend swapped from
# SerpAPI (DuckDuckGo) to ddgs' auto backend (rotates across engines) and fetch output capped.

glob FETCH_HEADERS: dict[str, str] = {   # BFCL's browser-like headers, minus "br" (no brotli decoder installed)
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Referer": "https://www.google.com/",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-User": "?1",
    "Sec-Fetch-Dest": "document"
};

# one tool call, rendered to text the way BFCL's executor renders it, cached by tool name + arguments
def call_web_tool(name: str, args: dict[str, str | int]) -> str {
    call = f"{name}:{json.dumps(args, sort_keys=True)}";
    CALLS.append(call);
    path = os.path.join(CACHE_DIR, hashlib.md5(call.encode()).hexdigest() + ".txt");
    if os.path.exists(path) {
        with open(path) as f {
            return f.read();
        }
    }
    try {
        result = bfcl_search_engine_query(**args) if name == "search_engine_query" else bfcl_fetch_url_content(**args);
    } except Exception as e {
        return f"Error during execution: {e}";
    }
    text = result if isinstance(result, str) else (json.dumps(result) if isinstance(result, dict) else str(result));
    if not (isinstance(result, dict) and "error" in result) {
        os.makedirs(CACHE_DIR, exist_ok=True);
        with open(path, "w") as f {
            f.write(text);
        }
    }
    return text;
}

glob RATE_LIMITED: list[str] = [];   # engines that answered HTTP 429 during the current search

# ddgs treats every non-200 page as "no results"; note the 429s so that a search left empty by
# rate limits is retried instead of reaching the agent as "No results found."
def engine_request(engine: BaseSearchEngine, *args: object, **kwargs: object) -> str | None {
    resp = engine.http_client.request(*args, **kwargs);
    if resp.status_code == 429 {
        RATE_LIMITED.append(engine.name);
    }
    return resp.text if resp.status_code == 200 else None;
}

with entry {
    setattr(BaseSearchEngine, "request", engine_request);   # jac check rejects assigning to a method attribute
}

# engines block bursts of searches: keep live ones MIN_SEARCH_INTERVAL_S apart across every run and arm
def wait_for_search_slot() -> None {
    last = 0.0;
    try {
        with open(SEARCH_CLOCK) as f {
            last = float(f.read());
        }
    } except (OSError, ValueError) {
        last = 0.0;
    }
    time.sleep(max(0.0, last + MIN_SEARCH_INTERVAL_S - time.time()));
    os.makedirs(CACHE_DIR, exist_ok=True);
    with open(SEARCH_CLOCK, "w") as f {
        f.write(str(time.time()));
    }
}

def bfcl_search_engine_query(keywords: str, max_results: int = 10, region: str = "wt-wt") -> list[dict[str, str]] | dict[str, str] {
    backoff = 60;
    hits: list[dict[str, str]] = [];
    for attempt in range(SEARCH_RETRIES + 1) {
        wait_for_search_slot();
        RATE_LIMITED.clear();
        try {
            hits = DDGS().text(keywords, region=region, max_results=max_results, backend="auto") as list[dict[str, str]];
            break;
        } except Exception as e {
            if not RATE_LIMITED {
                return {"error": str(e)};
            }
            if attempt == SEARCH_RETRIES {
                # the search backend is down, not the agent: end the process (every framework would hand an
                # exception back to the model) so eval/run.py records an error and reruns the task
                engines = ", ".join(sorted(set(RATE_LIMITED)));
                print(f"search rate-limited: HTTP 429 from {engines} after {SEARCH_RETRIES} retries (query: {keywords})", file=sys.stderr, flush=True);
                os._exit(RATE_LIMIT_EXIT);
            }
            time.sleep(backoff + random.uniform(0, 10));
            backoff = min(backoff * 2, 900);
        }
    }
    if not hits {
        return {"error": "Failed to retrieve the search results from server. Please try again later."};
    }
    return [{"title": h["title"], "href": h["href"], "body": h["body"]} for h in hits[:max_results]];
}

def bfcl_fetch_url_content(url: str, mode: str = "raw") -> dict[str, str] {
    if not url.startswith(("http://", "https://")) {
        raise ValueError(f"Invalid URL: {url}");
    }
    try {
        response = requests.get(url, headers=FETCH_HEADERS, timeout=20, allow_redirects=True);
        response.raise_for_status();
        if mode == "raw" {
            content = response.text;
        } elif mode == "markdown" {
            content = html2text.HTML2Text().handle(response.text);
        } elif mode == "truncate" {
            soup = BeautifulSoup(response.text, "html.parser");
            for script_or_style in (soup(["script", "style"]) as list[Tag]) {
                script_or_style.extract();
            }
            content = soup.get_text(separator="\n", strip=True);
        } else {
            raise ValueError(f"Unsupported mode: {mode}");
        }
        return {"content": content[:MAX_CONTENT_CHARS]};
    } except Exception as e {
        return {"error": f"An error occurred while fetching {url}: {str(e)}"};
    }
}


def normalize(text: str) -> str {
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip();
}

def matches(answer: str, gold: list[str]) -> bool {
    a = normalize(answer);
    for g in gold {
        n = normalize(g);
        if n and (n == a or n in a) {
            return True;
        }
    }
    return False;
}

def load_jsonl(path: str) -> list[dict] {
    rows: list[dict] = [];
    with open(path) as f {
        for line in f {
            if line.strip() {
                rows.append(dict(json.loads(line)));
            }
        }
    }
    return rows;
}

def load_question(task_id: str) -> str {
    for row in load_jsonl(DATA) {
        if row["id"] == task_id {
            turns = list(row["question"]);
            first = list(turns[0]);
            return str(dict(first[0])["content"]);
        }
    }
    raise KeyError(f"{task_id} not in {DATA}");
}

def load_gold(task_id: str) -> list[str] {
    if os.path.exists(ANSWERS) {
        for row in load_jsonl(ANSWERS) {
            if row["id"] == task_id {
                return [str(g) for g in list(row["ground_truth"])];
            }
        }
    }
    return [];
}


with entry {
    task_id = sys.argv[1] if len(sys.argv) > 1 else "web_search_0";
    question = load_question(task_id);
    gold = load_gold(task_id);
    errors: list[str] = [];
    try {
        answer = solve(question);
    } except Exception as e {
        errors.append(f"solve failed: {e}");
        answer = "";
    }
    verdict = "UNGRADED" if not gold else ("PASSED" if matches(answer, gold) else "FAILED");
    print(f"Verdict: {verdict}");
    print(f"Answer: {answer}");
    print(f"Gold: {' | '.join(gold)}");
    print(f"Tool calls: {len(CALLS)}");
    if errors {
        print("Diagnostics:");
        for error in errors {
            print(f"- {error}");
        }
    }
}
