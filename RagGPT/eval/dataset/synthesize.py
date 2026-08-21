"""Synthesize the eval dataset from the bundled Jac docs (the ground truth).

Six categories, each item carrying a gold routing label; doc-grounded items
also carry gold answers / reference code validated with `jac check`:

  rag_qa (80)      question + gold answer from one doc section, verified answerable
  coding (40)      imperative task whose reference solution compiles
  debugging (40)   bug-injected snippet that provably fails `jac check`
  small_talk (20)  pure pleasantries -> QAChat
  off_topic (20)   non-Jac questions -> OffTopicChat
  multi_turn (20)  two-turn scripts whose 2nd turn is unroutable without history

Resumable: appends to dataset.jsonl incrementally and only fills per-category
deficits on rerun. Synthesis/verification uses gpt-4.1 directly against
api.openai.com (never through the eval proxy).

Run:  /home/xiaoyu/miniconda3/envs/jaseci/bin/python synthesize.py [--smoke]
"""

import argparse
import random
import re
import sys
import threading
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common  # noqa: E402

DOCS_DIR = common.CODER_DIR / "Jac-Rag-GPT" / "docs"

POOL = [
    "reference/language/access-modifiers.md",
    "reference/language/advanced.md",
    "reference/language/concurrency.md",
    "reference/language/foundation.md",
    "reference/language/functions-objects.md",
    "reference/language/library-mode.md",
    "reference/language/osp.md",
    "reference/language/primitives.md",
    "reference/language/python-integration.md",
    "reference/language/walker-responses.md",
    "reference/plugins/byllm.md",
    "reference/persistence.md",
    "reference/code-organization.md",
    "reference/testing.md",
    "tutorials/language/basics.md",
    "tutorials/language/coding_primer.md",
    "tutorials/language/debugging.md",
    "tutorials/language/osp.md",
    "tutorials/ai/quickstart.md",
    "tutorials/ai/structured-outputs.md",
    "tutorials/ai/agentic.md",
    "tutorials/ai/multimodal.md",
    "internals/abstractions.md",
    "internals/compiler_architecture.md",
    "internals/interop.md",
    "internals/jac_import_patterns.md",
]

TARGETS = {
    "rag_qa": 80, "coding": 40, "debugging": 40,
    "small_talk": 20, "off_topic": 20, "multi_turn": 20,
}

GOLD_AGENT = {
    "rag_qa": "RagChat", "coding": "CodingChat", "debugging": "DebuggerChat",
    "small_talk": "QAChat", "off_topic": "OffTopicChat",
}

MAX_PER_FILE = {"rag_qa": 5, "coding": 3, "debugging": 3}
SECTION_MIN, SECTION_MAX = 400, 2500
WORKERS = 6

DEBUG_TEMPLATES = [
    "This Jac code gives an error, please fix it:\n```jac\n{code}\n```",
    "Why doesn't this Jac code compile? Can you fix it?\n```jac\n{code}\n```",
    "I'm getting an error with this Jac code, help me debug it:\n```jac\n{code}\n```",
    "My Jac program is broken and I can't figure out why:\n```jac\n{code}\n```",
    "Something is wrong with this Jac snippet, it won't run:\n```jac\n{code}\n```",
]

JAC_STYLE_HINT = (
    "The program must be valid Jac 0.31 (NOT Python): braces not indentation, every "
    "statement ends with ';', `with entry { ... }` as the entry block, archetypes "
    "declared as `node`/`edge`/`walker`/`obj`, fields with `has name: type;`. If the "
    "program uses an LLM, use exactly:\n"
    'import from jaclang.byllm.lib { Model }\nglob llm = Model(model_name="gpt-4.1-mini");\n'
    "and `def fn(...) -> T by llm();`. Do not import any other external package."
)


class QAGen(BaseModel):
    question: str
    gold_answer: str


class QAVerify(BaseModel):
    answerable: bool
    gold_correct: bool
    too_generic: bool
    reason: str


class CodingGen(BaseModel):
    task: str
    reference_solution: str


class DebugGen(BaseModel):
    clean_program: str
    buggy_code: str
    bug_description: str


class MsgList(BaseModel):
    messages: list[str]


class Turn2Gen(BaseModel):
    message: str


client: OpenAI | None = None

_lock = threading.Lock()
_stats: dict[str, int] = {}


def bump(key: str) -> None:
    with _lock:
        _stats[key] = _stats.get(key, 0) + 1


def llm(messages: list[dict], schema: type[BaseModel], temperature: float = 0.7):
    global client
    if client is None:
        common.require_api_key()
        client = OpenAI()  # honours $OPENAI_BASE_URL; still outside the token proxy
    parse = getattr(client.chat.completions, "parse", None) or client.beta.chat.completions.parse
    resp = parse(model=common.JUDGE_MODEL, messages=messages,
                 response_format=schema, temperature=temperature)
    return resp.choices[0].message.parsed


def split_sections(rel_path: str) -> list[dict]:
    """Split a doc file into heading-delimited sections of SECTION_MIN..SECTION_MAX chars."""
    text = (DOCS_DIR / rel_path).read_text(encoding="utf-8", errors="replace")
    parts = re.split(r"(?m)^(#{1,4} .+)$", text)
    # parts = [preamble, heading, body, heading, body, ...]
    raw = []
    if parts[0].strip():
        raw.append(("(intro)", parts[0]))
    for i in range(1, len(parts) - 1, 2):
        raw.append((parts[i].strip(), parts[i + 1]))

    sections, buf_head, buf = [], None, ""
    for head, body in raw:
        chunk = f"{head}\n{body}".strip()
        if buf and len(buf) < SECTION_MIN:
            buf += "\n\n" + chunk
        else:
            if buf:
                sections.append((buf_head, buf))
            buf_head, buf = head, chunk
    if buf:
        sections.append((buf_head, buf))

    out = []
    for head, content in sections:
        if len(content) < SECTION_MIN:
            continue
        if len(content) > SECTION_MAX:
            cut = content.rfind("\n\n", 0, SECTION_MAX)
            content = content[: cut if cut > SECTION_MIN else SECTION_MAX]
        out.append({"source_file": rel_path, "source_section": head, "content": content})
    return out


def load_existing() -> list[dict]:
    return common.read_jsonl(common.DATASET_PATH)


def next_id(existing_ids: set, category: str) -> str:
    n = 1
    while f"{category}-{n:03d}" in existing_ids:
        n += 1
    return f"{category}-{n:03d}"


def save_item(item: dict, existing_ids: set) -> None:
    with _lock:
        item["id"] = next_id(existing_ids, item["category"])
        existing_ids.add(item["id"])
        common.append_jsonl(common.DATASET_PATH, item)


# ---------------------------------------------------------------- rag_qa

def gen_rag_qa(section: dict) -> dict | None:
    gen = llm([
        {"role": "system", "content":
            "You create eval questions for a chatbot that answers questions about the Jac "
            "programming language. From the documentation section provided, write ONE natural "
            "question a developer might ask in chat, answerable from this section ALONE, and a "
            "concise gold answer (2-6 sentences) based strictly on the section. The question "
            "must be specific — name concrete Jac constructs, keywords, or behaviors from the "
            "section — and must NOT reference 'the docs', 'this section', or similar. It should "
            "be a conceptual question (what/how/why/when), not a request to write or fix code."},
        {"role": "user", "content":
            f"File: {section['source_file']}\nSection: {section['source_section']}\n\n"
            f"{section['content']}"},
    ], QAGen)
    if gen is None:
        bump("rag_qa_gen_failed")
        return None

    verdict = llm([
        {"role": "system", "content":
            "You are auditing an eval item for a Jac-language QA dataset. Judge strictly, "
            "using ONLY the provided documentation section as ground truth."},
        {"role": "user", "content":
            f"Documentation section:\n{section['content']}\n\n"
            f"Question: {gen.question}\n\nProposed gold answer: {gen.gold_answer}\n\n"
            "Assess: (1) answerable — can the question be fully answered from this section "
            "alone? (2) gold_correct — is the gold answer correct and complete per the "
            "section? (3) too_generic — could this question be asked of almost any "
            "programming language with a trivially generic answer?"},
    ], QAVerify, temperature=0.0)
    if verdict is None or not (verdict.answerable and verdict.gold_correct) or verdict.too_generic:
        bump("rag_qa_rejected")
        return None

    return {
        "category": "rag_qa",
        "turns": [{"message": gen.question, "gold_agent": "RagChat"}],
        "gold_answer": gen.gold_answer,
        "source_file": section["source_file"],
        "source_section": section["source_section"],
    }


# ---------------------------------------------------------------- coding

def gen_coding(section: dict) -> dict | None:
    messages = [
        {"role": "system", "content":
            "You create coding tasks for a Jac-language eval dataset. From the documentation "
            "section provided, write ONE imperative coding request as a chat message ('Write a "
            "Jac walker that ...', 'Create a Jac program that ...'), solvable in under 40 lines, "
            "exercising the Jac feature the section teaches. Also provide reference_solution: a "
            "complete standalone Jac program that solves the task. " + JAC_STYLE_HINT +
            " The task must not mention 'the docs' or 'this section'."},
        {"role": "user", "content":
            f"File: {section['source_file']}\nSection: {section['source_section']}\n\n"
            f"{section['content']}"},
    ]
    for _ in range(3):
        gen = llm(messages, CodingGen)
        if gen is None:
            bump("coding_gen_failed")
            return None
        ok, out = common.jac_check(gen.reference_solution)
        if ok:
            return {
                "category": "coding",
                "turns": [{"message": gen.task, "gold_agent": "CodingChat"}],
                "reference_code": gen.reference_solution,
                "source_file": section["source_file"],
                "source_section": section["source_section"],
            }
        bump("coding_reference_check_failed")
        messages.append({"role": "assistant", "content": gen.model_dump_json()})
        messages.append({"role": "user", "content":
            f"Your reference_solution failed `jac check`:\n{out[-2000:]}\n"
            "Fix the reference so it compiles, and keep the task consistent with it."})
    bump("coding_rejected")
    return None


# ---------------------------------------------------------------- debugging

def gen_debugging(section: dict, template: str) -> dict | None:
    messages = [
        {"role": "system", "content":
            "You create debugging tasks for a Jac-language eval dataset. From the documentation "
            "section provided: (1) write clean_program — a complete standalone Jac program "
            "(under 35 lines) using the section's feature, which compiles; (2) write buggy_code — "
            "an exact copy with EXACTLY ONE injected bug that makes compilation or type checking "
            "fail (e.g. wrong keyword, missing ';', wrong operator, bad type annotation, "
            "misspelled attribute, wrong archetype syntax); (3) bug_description — one sentence "
            "naming the bug. " + JAC_STYLE_HINT},
        {"role": "user", "content":
            f"File: {section['source_file']}\nSection: {section['source_section']}\n\n"
            f"{section['content']}"},
    ]
    for _ in range(3):
        gen = llm(messages, DebugGen)
        if gen is None:
            bump("debugging_gen_failed")
            return None
        clean_ok, clean_out = common.jac_check(gen.clean_program)
        buggy_ok, _ = common.jac_check(gen.buggy_code)
        if clean_ok and not buggy_ok:
            return {
                "category": "debugging",
                "turns": [{"message": template.format(code=gen.buggy_code),
                           "gold_agent": "DebuggerChat"}],
                "reference_code": gen.clean_program,
                "buggy_code": gen.buggy_code,
                "bug_description": gen.bug_description,
                "source_file": section["source_file"],
                "source_section": section["source_section"],
            }
        bump("debugging_check_failed")
        problem = (
            f"clean_program failed `jac check`:\n{clean_out[-2000:]}" if not clean_ok
            else "buggy_code still PASSES `jac check` — the injected bug must make it fail."
        )
        messages.append({"role": "assistant", "content": gen.model_dump_json()})
        messages.append({"role": "user", "content": problem + "\nRegenerate all three fields."})
    bump("debugging_rejected")
    return None


# ---------------------------------------------------------------- small talk / off topic

def gen_message_list(category: str, count: int) -> list[str]:
    prompts = {
        "small_talk": (
            f"Generate {count} distinct short chat messages of pure small talk directed at an "
            "assistant: greetings, thanks, farewells, 'how are you', pleasantries. No technical "
            "content whatsoever, no questions about any tool or language. Vary tone, length "
            "(2-15 words), and punctuation."),
        "off_topic": (
            f"Generate {count} distinct chat messages a user might wrongly send to a "
            "Jac-language assistant: questions about OTHER programming languages (Python, "
            "JavaScript, Rust...), general tech (docker, git, databases), or non-programming "
            "topics (cooking, travel, sports). NEVER mention Jac or Jaseci. Vary style and "
            "length; make most of them questions."),
    }
    gen = llm([{"role": "user", "content": prompts[category]}], MsgList, temperature=1.0)
    return gen.messages[:count] if gen else []


# ---------------------------------------------------------------- multi turn

TURN2_TARGETS = (["CodingChat"] * 6 + ["RagChat"] * 5 + ["DebuggerChat"] * 5 + ["QAChat"] * 4)

TURN2_GUIDE = {
    "CodingChat": "a request to write, extend, or modify code based on what was just discussed "
                  "(e.g. 'can you turn that into code?', 'now make it recursive')",
    "RagChat": "a conceptual follow-up question about why/how the thing just discussed works "
               "(e.g. 'why does it work that way?', 'when would I use that instead?')",
    "DebuggerChat": "a report that the thing just provided fails when tried "
                    "(e.g. 'I ran it and got an error, can you fix it?')",
    "QAChat": "pure closing pleasantry with zero technical content (e.g. 'thanks, that helped!')",
}


def gen_multi_turn(source_item: dict, target_agent: str) -> dict | None:
    t1 = source_item["turns"][0]["message"]
    context = source_item.get("gold_answer") or source_item.get("reference_code") or ""
    gen = llm([
        {"role": "system", "content":
            "You write the SECOND user turn of a two-turn chat eval item. The second turn must "
            "be short (under 20 words), must read naturally after the first turn, and must be "
            "IMPOSSIBLE to route correctly without seeing the first turn: use pronouns "
            "('it', 'that') instead of naming Jac, Jaseci, or any construct from turn one. "
            f"The second turn must be {TURN2_GUIDE[target_agent]}."},
        {"role": "user", "content":
            f"First user turn:\n{t1[:1500]}\n\n"
            f"(The assistant answered it roughly as follows: {context[:800]})\n\n"
            "Write the second user turn."},
    ], Turn2Gen)
    if gen is None or len(gen.message) > 200:
        bump("multi_turn_rejected")
        return None
    low = gen.message.lower()
    if "jac" in low or "jaseci" in low:
        bump("multi_turn_rejected")
        return None
    return {
        "category": "multi_turn",
        "turns": [
            {"message": t1, "gold_agent": source_item["turns"][0]["gold_agent"]},
            {"message": gen.message, "gold_agent": target_agent},
        ],
        "gold_answer": source_item.get("gold_answer"),
        "reference_code": source_item.get("reference_code"),
        "source_file": source_item.get("source_file"),
        "meta": {"turn1_from": source_item["id"], "turn2_target": target_agent},
    }


# ---------------------------------------------------------------- orchestration

def fill_doc_category(category: str, deficit: int, sections: list[dict],
                      existing: list[dict], existing_ids: set) -> None:
    from concurrent.futures import ThreadPoolExecutor

    used_sections = {(it.get("source_file"), it.get("source_section"))
                     for it in existing if it["category"] == category}
    per_file: dict[str, int] = {}
    for it in existing:
        if it["category"] == category:
            per_file[it["source_file"]] = per_file.get(it["source_file"], 0) + 1

    candidates = [s for s in sections
                  if (s["source_file"], s["source_section"]) not in used_sections]
    if category in ("coding", "debugging"):
        candidates = [s for s in candidates if "```jac" in s["content"]]
    rng = random.Random(42 if category == "rag_qa" else 43 if category == "coding" else 44)
    rng.shuffle(candidates)

    done = threading.Event()
    accepted = [0]
    idx = [0]

    def worker(wid: int) -> None:
        while not done.is_set():
            with _lock:
                if accepted[0] >= deficit or idx[0] >= len(candidates):
                    done.set()
                    return
                section = candidates[idx[0]]
                idx[0] += 1
                if per_file.get(section["source_file"], 0) >= MAX_PER_FILE[category]:
                    continue
            if category == "rag_qa":
                item = gen_rag_qa(section)
            elif category == "coding":
                item = gen_coding(section)
            else:
                item = gen_debugging(section, DEBUG_TEMPLATES[wid % len(DEBUG_TEMPLATES)])
            if item is None:
                continue
            with _lock:
                if accepted[0] >= deficit:
                    done.set()
                    return
                accepted[0] += 1
                per_file[section["source_file"]] = per_file.get(section["source_file"], 0) + 1
            save_item(item, existing_ids)
            print(f"  [{category}] {item['id']}  <- {item['source_file']}", flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(worker, range(WORKERS)))
    if accepted[0] < deficit:
        print(f"  WARNING: {category} exhausted candidates at {accepted[0]}/{deficit}")


def write_report() -> None:
    items = load_existing()
    by_cat: dict[str, list[dict]] = {}
    for it in items:
        by_cat.setdefault(it["category"], []).append(it)
    lines = ["# Eval dataset report", "",
             f"Total items: {len(items)}", "", "| category | count | target |",
             "|---|---|---|"]
    for cat, target in TARGETS.items():
        lines.append(f"| {cat} | {len(by_cat.get(cat, []))} | {target} |")
    lines += ["", "## Synthesis rejection stats", ""]
    for key in sorted(_stats):
        lines.append(f"- {key}: {_stats[key]}")
    lines += ["", "## Source-file spread (doc-grounded categories)", ""]
    spread: dict[str, int] = {}
    for it in items:
        if it.get("source_file"):
            spread[it["source_file"]] = spread.get(it["source_file"], 0) + 1
    for f, n in sorted(spread.items(), key=lambda kv: -kv[1]):
        lines.append(f"- {f}: {n}")
    lines += ["", "## One example per category", ""]
    for cat in TARGETS:
        if by_cat.get(cat):
            ex = by_cat[cat][0]
            msg = ex["turns"][0]["message"].replace("\n", " ")[:200]
            lines.append(f"**{cat}** ({ex['id']}): {msg}")
            lines.append("")
    common.DATASET_REPORT_PATH.write_text("\n".join(lines))
    print(f"report -> {common.DATASET_REPORT_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny targets (3 per category)")
    args = ap.parse_args()

    targets = {k: min(3, v) for k, v in TARGETS.items()} if args.smoke else TARGETS

    existing = load_existing()
    existing_ids = {it["id"] for it in existing}
    counts = {cat: sum(1 for it in existing if it["category"] == cat) for cat in TARGETS}
    print(f"existing: {counts}")

    sections = []
    for rel in POOL:
        if (DOCS_DIR / rel).exists():
            sections.extend(split_sections(rel))
        else:
            print(f"  WARNING: pool file missing: {rel}")
    print(f"section pool: {len(sections)} sections from {len(POOL)} files")

    for cat in ("rag_qa", "coding", "debugging"):
        deficit = targets[cat] - counts[cat]
        if deficit > 0:
            print(f"generating {deficit} x {cat} ...")
            fill_doc_category(cat, deficit, sections, existing, existing_ids)
            existing = load_existing()

    for cat in ("small_talk", "off_topic"):
        deficit = targets[cat] - counts[cat]
        if deficit > 0:
            print(f"generating {deficit} x {cat} ...")
            for msg in gen_message_list(cat, deficit):
                save_item({"category": cat,
                           "turns": [{"message": msg, "gold_agent": GOLD_AGENT[cat]}]},
                          existing_ids)
            existing = load_existing()

    deficit = targets["multi_turn"] - counts["multi_turn"]
    if deficit > 0:
        print(f"generating {deficit} x multi_turn ...")
        used = {it.get("meta", {}).get("turn1_from") for it in existing
                if it["category"] == "multi_turn"}
        rng = random.Random(45)
        pool_items = [it for it in existing
                      if it["category"] in ("rag_qa", "coding", "debugging")
                      and it["id"] not in used]
        rng.shuffle(pool_items)
        targets_cycle = (TURN2_TARGETS * (deficit // len(TURN2_TARGETS) + 1))[:deficit]
        rng.shuffle(targets_cycle)
        made = 0
        for src in pool_items:
            if made >= deficit:
                break
            item = gen_multi_turn(src, targets_cycle[made % len(targets_cycle)])
            if item:
                save_item(item, existing_ids)
                print(f"  [multi_turn] {item['id']}  t2->{item['meta']['turn2_target']}")
                made += 1

    write_report()


if __name__ == "__main__":
    main()
