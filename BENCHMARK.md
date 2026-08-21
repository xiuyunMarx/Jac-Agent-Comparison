# Running the comparison

Five benchmarks, each implementing the same agent three ways, all against one
locally served model:

| benchmark | task | Jac/byLLM | agent framework | no framework |
|---|---|---|---|---|
| `CodeAgent` | SWE-bench Lite, 36 instances | `byLLM/` | LangGraph | `openai_sdk/` |
| `RagGPT` | routed RAG over the Jac docs, 220 items | `Jac-Rag-GPT/` (+ router variant) | LangGraph | `openai_sdk/` |
| `YTNavigator` | grounded QA over a synthetic channel, 23 questions | `byLLM/` | LangGraph (Django) | `openai_sdk/` |
| `meeting-assistant` | transcript to task list, 10 transcripts | `byLLM/` | CrewAI | `openai_sdk/` |
| `Email-Auto-response` | triage and draft replies, 6 mailboxes | `byLLM/` | CrewAI + LangGraph | `openai_sdk/` |

```bash
./run_benchmark.sh --verify-only   # seconds: checks the wiring, calls no model
./run_benchmark.sh --smoke         # minutes: a few cases per benchmark
./run_benchmark.sh --clean         # delete old results, then the full sweep
```

Everything lands in `bench/runs/<run-id>/`: one log per step, a `manifest.json`
of what ran, `tokens.jsonl` (every call from every arm), and `summary.md`.

## How it fits together

`run_benchmark.sh` is a thin wrapper. The work is in `bench/`:

| | |
|---|---|
| `bench/config.py` | the model seam: one endpoint, fifteen arms |
| `bench/services.py` | ollama, the token proxy, and a capability probe |
| `bench/verify.py` | proves the wiring without spending a token |
| `bench/clean.py` | delete every result, keep every input |
| `bench/run_all.py` | the sweep, stage by stage |
| `bench/report.py` | one table across all five benchmarks |

None of it reimplements a benchmark. Each project already has a runner that
knows how to drive its own arms -- `swebench_bridge/compare.py` pins one
instance set across three frameworks, `YTNavigator/eval/e2e.py` brings up a
database first, `RagGPT/eval/harness/run_eval.py` resumes a half-finished
sweep. `run_all.py` decides what runs in what order and hands each runner the
environment that points it at the shared model.

**Batching here means one shared server, not concurrent requests.** All fifteen
arms talk to a single long-lived endpoint. Within a stage the arms run one at a
time on purpose: they are being compared on tokens and wall-clock, and two arms
in flight would contend for the same GPU and corrupt both numbers.

## Verifying without running anything

`bench/verify.py` answers "would this work?" without calling a model or loading
weights -- which matters, because the expensive failures here are not inference
failures. They are two arms of one benchmark quietly resolving to different
models, a runner that never learned about its third arm, an absolute path that
only exists on one machine. All of those are visible statically.

```bash
python -m bench.verify            # 47 checks
python -m bench.verify --models   # just the fifteen model seams
```

It reports, per arm, the model id that arm would actually put on the wire:

```
  PASS  CodeAgent    langgraph     [import]  muse-glimmer | stream=False | max_tokens=16384
  PASS  meeting      CrewAI        [import]  openai/muse-glimmer
  PASS  YTNavigator  YT-Navigator  [import]  muse-glimmer via ChatOpenAI
  PASS  16/16 arms resolve to 'muse-glimmer' (in each library's own shape)
```

Python arms are checked live -- the module is imported with the benchmark
environment and asked what it resolved, which costs nothing because
constructing a client makes no request. Jac arms are checked statically
(`jac check` plus the presence of the env knob), because importing one builds
its whole graph and for two of them that means loading an embedding model. Each
row says which kind of check produced it.

It also confirms every runner offers all three of its arms, the datasets are
present at their expected sizes, the toolchain and dependencies are importable,
and no hardcoded absolute path survived.

## The model

One knob reaches every arm:

```bash
BENCH_MODEL=muse-glimmer ./run_benchmark.sh
```

Three libraries stand between the arms and the model, and each wants the id in
a different shape -- `openai/x` for litellm (byLLM, CrewAI), `openai:x` for
langchain's `init_chat_model`, bare `x` for the raw SDK. `bench/config.py`
emits the right shape per project, and each arm normalizes whatever it gets.
That is why one export cannot put two arms of the same benchmark on different
models, which used to be possible and is the failure mode that makes a
benchmark measure the model instead of the framework.

Other knobs, all optional:

| variable | default | |
|---|---|---|
| `BENCH_MODEL` | `muse-glimmer` | model to pull and serve |
| `BENCH_BASE_URL` | `http://127.0.0.1:11434/v1` | any OpenAI-compatible endpoint |
| `BENCH_CTX` | `32768` | context window baked into the served model |
| `BENCH_JUDGE_MODEL` | the same model | LLM-as-judge |
| `BENCH_CONDA_ENV` | `jaseci` | environment to run in |
| `BENCH_NO_PROXY` | unset | skip the token ledger |
| `CODEAGENT_TEMPERATURE` | `1.0` | all three CodeAgent arms; 1.0 is what gpt-5 forced |

### Context window

Ollama defaults `num_ctx` to 4096 and will not take it over the `/v1`
endpoint, while CodeAgent alone asks for a 16k completion. So `services.py`
derives `<model>-bench32k` with `ollama create` and serves that. Change
`BENCH_CTX` and it derives a different one.

### The probe

Before anything expensive, `run_benchmark.sh` checks the served model reports
token usage, emits `tool_calls`, and honours a strict JSON schema. All five
benchmarks drive the model with tool calls; one that cannot produces a table of
zeros after several hours that reads like a framework result and is not one.
The sweep refuses to start if the probe fails.

```bash
./run_benchmark.sh --probe-only
```

A model that passes tool calling but fails strict JSON schema will still
complete the run -- every arm has a text fallback. Watch the `parse-fail`
column in the summary.

## The token ledger

Every arm runs with `OPENAI_BASE_URL` pointed at
`RagGPT/eval/harness/proxy.py`, which forwards to the model server and logs
each call with its usage. byLLM, LangGraph, CrewAI and the raw SDK are
therefore all measured the same way, by something none of them can talk its way
around, and `bench/report.py` reconciles that total against each project's own
accounting. A gap means an arm found a way past the instrumentation.

Two caveats against a local server. `prompt_tokens_details.cached_tokens` and
`completion_tokens_details.reasoning_tokens` are OpenAI-only, so cache and
reasoning columns read zero rather than being wrong. And streaming carries
usage only via `stream_options.include_usage`, which not every server accepts
-- so `CODEAGENT_STREAM` defaults to off locally. Set `BENCH_STREAM=1` to
restore it.

## The judge

`--judge` is on by default and runs on the same local model, so the sweep needs
no API key at all. The report labels those columns, because a model grading its
own family is a biased grader: they compare arms against each other, not
against an absolute. `--no-judge` keeps only deterministic metrics.

## Moving to another machine

0. **Clone with submodules.** `CodeAgent/SWE-bench` is the official grading
   harness, vendored as a submodule pinned to its upstream. A plain `git clone`
   leaves that directory empty, and `pip install -e CodeAgent/SWE-bench` then
   fails with *"does not appear to be a Python project: neither 'setup.py' nor
   'pyproject.toml' found"*.
   ```bash
   git clone --recurse-submodules https://github.com/xiuyunMarx/Jac-Agent-Comparison.git
   # already cloned without it:
   git submodule update --init --recursive
   ```
1. **Python.** 3.13.x -- not 3.14 (the CrewAI arms cap there), not below 3.13.2
   (YT-Navigator's floor). Create the environment first: installing into a
   system Python gives `Defaulting to user installation because normal
   site-packages is not writeable`, and the pins here will fight whatever the
   system already has.
   ```bash
   conda create -n jaseci python=3.13 && conda activate jaseci
   pip install -r requirements.txt
   pip install -e CodeAgent/SWE-bench
   ```
   `requirements.txt` deliberately does **not** list `jaclang` -- see step 3.
2. **The Email CrewAI arm's own venv.** It pins langgraph 1.x, which cannot
   coexist with the 0.3.5 two other benchmarks need.
   ```bash
   cd Email-Auto-response/CrewAI-LangGraph
   python -m venv .venv && .venv/bin/pip install -r ../../requirements-email-crewai.txt
   ```
3. **Jac.** `jaclang` is not pip-installable at the version this was built
   against and is not in `requirements.txt`: the toolchain ships its own copy,
   and `pip install jaclang==<that version>` fails with *"Could not find a
   version that satisfies the requirement"* because PyPI's `jaclang` is a
   different, older lineage. Install the Jac toolchain itself and let it
   provide the runtime.

   Then run `jac install` once in each Jac project (`CodeAgent/byLLM`,
   `Email-Auto-response/byLLM`, `meeting-assistant/byLLM`, `YTNavigator/byLLM`,
   `RagGPT/Jac-Rag-GPT`, `RagGPT/Jac-Rag-GPT-ByllmRouter`). Packages
   pip-installed in step 1 are **not** visible to the Jac runtime.

   The eval uses whatever `jac` is on `PATH`; no path is hardcoded anywhere. If
   the binary is somewhere unusual, export `JAC_BIN=<the directory holding it>`.

   `python -m bench.verify` reports which `jac` it found, its version, and
   whether byLLM resolves under it -- byLLM lives at `jaclang.byllm` in a
   toolchain that bundles it and as the separate `byllm` package alongside a
   released `jaclang`, and the Jac arms name one of the two. If your runtime
   provides the other, verify says so and names the files.
4. **The model.** `ollama pull $BENCH_MODEL`, or point `BENCH_BASE_URL` at a
   vLLM/llama.cpp server you already run.
5. **Postgres with pgvector**, for YTNavigator. Its eval starts one itself from
   any conda env that has the binaries:
   ```bash
   conda create -y -n ytnav-pg -c conda-forge postgresql pgvector
   ```
   `eval/pgdata` is deliberately not copied between machines -- a Postgres data
   directory is not portable, and `e2e.py` rebuilds it from
   `YTNavigator/datasets/`.
6. **A container runtime**, for CodeAgent: `docker`, or `pip install udocker &&
   udocker install`. Budget 1-2 GB per SWE-bench instance image and keep 6 GB
   free per worker; the first run pulls 36 images.
7. **Network on the first run.** `datasets.load_dataset("SWE-bench/SWE-bench_Lite")`
   and the SWE-bench images both need it. Everything else is offline.

Then:

```bash
./run_benchmark.sh --verify-only    # is the wiring right? (no model needed)
./run_benchmark.sh --probe-only     # is the model up to it?
./run_benchmark.sh --smoke          # does a small run come out clean?
./run_benchmark.sh --clean          # the real thing
```

Run `--verify-only` first: it needs no weights, so on a fresh machine its
failures are the ones worth seeing before an 18 GB download.

## What gets deleted

`bench/clean.py --dry-run` prints the manifest before touching anything. It
removes every result and keeps every input, including the ones that live in an
output directory and do not look like inputs:

- `Email-Auto-response/eval/out/benchmark_slide.pptx` is hand-made; nothing
  regenerates it, so it is moved aside and restored.
- `RagGPT/eval/dataset/dataset.jsonl` was synthesized with gpt-4.1. It cannot
  be rebuilt offline, and rebuilding it would change the benchmark rather than
  re-run it.
- `meeting-assistant/byLLM/meeting_notes.txt` looks like leftover output and is
  the transcript that arm reads when run standalone.
- `CodeAgent/case_study/instances.txt` is generated, but it is the pinned
  instance set the whole comparison rests on.
- The `faiss_index/` directories are deterministic derived data; rebuilding
  them would add an uncontrolled variable to a retrieval comparison.

`YTNavigator/eval/pgdata` **is** removed -- it is a Postgres data directory,
not a result, and `e2e.py` rebuilds it. A live postmaster is stopped first.

## Reading the summary

`bench/report.py` puts each benchmark's own headline metric next to the tokens
that bought it. There is no combined score: a resolve rate and a routing
accuracy do not average into anything meaningful.

It also asserts **three-arm coverage** and exits non-zero on an empty cell. That
check exists because four of these five benchmarks sat for a long time with a
written-but-never-executed OpenAI-SDK arm, and nothing in the tooling ever said
so.
