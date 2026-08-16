# Coding agents ⟶ SWE-bench

Runs any of the coding-agent implementations over SWE-bench instances and grades
the result with the official harness vendored at `../SWE-bench`. The three are
one agent expressed three ways, and the comparison between them is the point:

| framework | agent | what it is |
|---|---|---|
| `byllm` | [`../byLLM`](../byLLM) | Jac + byLLM: a walker over a phase graph |
| `langgraph` | [`../langgraph`](../langgraph) | Python + LangGraph: a compiled `StateGraph` |
| `openai` | [`../openai_sdk`](../openai_sdk) | Python, no framework: a `while` loop over the OpenAI SDK |

```
frameworks.py    the registry -- the whole of the fork between implementations
runtime.py       docker / udocker: getting an image here and running commands in it
workspace.py     the objective, the preparation step, and patch extraction
run_agent.py     inference: instances -> predictions.jsonl (+ runs.jsonl)
grade.py         grading without a Docker daemon: predictions -> the harness verdict
report.py        one run summarized, or any number of them compared
compare.py       every framework over one pinned instance set, graded, side by side
swe_entry.jac    the Jac shim for one instance
swe_entry.py     the Python shim for one instance, shared by langgraph and openai
```

## Any number of implementations

Nothing here counts frameworks. `compare.py --frameworks` takes as many as you
name, `report.py` takes as many run directories as you give it, and the
comparison it prints is a **partition**: every graded instance is filed under
exactly which subset of the implementations resolved it.

```
  Agreement over 30 instance(s) run by all 3
    resolved by all          : 7
    byllm + langgraph        : 3
    byllm + openai           : 1
    langgraph + openai       : 2
    only byllm               : 4
    only langgraph           : 2
    only openai              : 1
    resolved by none         : 10
```

Two resolve rates cannot tell you whether the implementations solved the same
problems, and with three sides the gap is worse: three runs at 40% could be the
same twelve instances or thirty-six different ones. Adding a fourth
implementation is one entry in `frameworks.py`.

## Setup

1. **A container runtime.** Either the Docker daemon reachable as your own user:

   ```bash
   sudo usermod -aG docker $USER    # then open a new login shell
   docker ps                        # must succeed with no sudo
   ```

   or, with no sudo and no group membership, udocker — which is the default:

   ```bash
   pip install udocker && udocker install
   ```

2. **The harness package**, for grading:

   ```bash
   pip install -e ../SWE-bench
   ```

3. **A provider key** for the model the agents call:

   ```bash
   export OPENAI_API_KEY=sk-...
   ```

4. **The runner for whichever framework you are using.** Neither project needs
   installing — each shim puts its agent home on `sys.path` itself — but `jac`
   must be on PATH for the byLLM side, and the interpreter given by `--python`
   (default: the one running the driver) needs the Python side's dependencies:

   ```bash
   pip install langgraph langchain-core langchain-openai pydantic   # langgraph
   pip install openai                                               # openai
   ```

   `run_agent.py` checks this for you: it loads the agent once at startup,
   before pulling any image or spawning any instance, and warns if it did not
   come up cleanly.

5. **Disk.** Each instance image is 1–2 GB and each workspace is a full repo
   copy. Both `run_agent.py` and `grade.py` hold a floor (`--min-free-gb`,
   default 6) that a worker waits behind before unpacking another container, and
   `grade.py` retires each image the moment its instance is decided. Grading 300
   instances without that is about 500 GB of images.

## Run one

```bash
python run_agent.py --framework openai --run-id smoke --limit 2
python grade.py --predictions results/smoke/predictions.jsonl
python report.py results/smoke
```

`run_agent.py` is resumable: it skips instances already in `predictions.jsonl`,
so an interrupted run continues by re-issuing the same command with the same
`--run-id`. `--force` re-runs them instead. `grade.py` is resumable the same way
through `eval_results.jsonl`, so an interrupted grade costs one instance.

## Run them all

```bash
python compare.py --run-id three-way \
    --frameworks byllm langgraph openai \
    --instances-file ../case_study/instances.txt \
    --workers 6 --eval-workers 2
```

Two things it does that running the commands by hand does not:

* **The instance set is resolved once and pinned.** Every side is handed the
  same explicit `--instance-ids`, so `--limit 20` cannot mean a different twenty
  on the second side. A comparison over different instance sets is not one.
* **The frameworks run one after another, never at once.** They would otherwise
  contend for the same cores, disk and runtime, and the wall-clock and token
  numbers are meant to be read against each other.

Any flag it does not recognise is forwarded to `run_agent.py`, so `--workers 8`,
`--max-steps 12`, `--network none` and the rest work here too. The flags that
must be identical on every side — `--dataset`, `--split`, `--repo`, `--limit`,
`--instance-ids`, `--instances-file`, `--run-id`, `--output-dir`, `--runtime` —
belong to `compare.py` itself and are refused in the passthrough.

Already have graded runs? Compare them without re-grading:

```bash
python report.py results/three-way-byllm results/three-way-langgraph results/three-way-openai
```

### Options worth knowing

| flag | why |
|---|---|
| `--framework` | which agent, and with it the shim, the agent home, the container prefix and the `model_name_or_path`. |
| `--runtime` | `udocker` (default) or `docker`. Picks how the image is obtained and where commands run, for inference *and* grading. |
| `--model` | the model the agent calls (default `gpt-5`). Pin it identically across the frameworks, or you are measuring the model. |
| `--max-steps` | phase budget for one agent run (default 10). The agents' own loop brakes are per-phase; this bounds the traversal. |
| `--workers` | instances in flight. Each holds one container and one agent process. |
| `--instance-timeout` | wall-clock ceiling per instance (default 1800s). |
| `--instances-file` | a file of instance ids, one per line — how a study pins its set. |
| `--min-free-gb` | the disk floor a worker waits behind before unpacking a container. |
| `--cleanup-images` | delete each instance image after inference; `grade.py` does this by default (`--keep-images` to stop it). |
| `--prepare` | `auto` (default) / `always` / `never` — see below. |
| `--network none` | cut the workspace container off the network. Some suites fail without it; most do not need it. |

### `--prepare`

Bind-mounting the workspace over `/testbed` is transparent only when the image
installed the repo *editably*, which points site-packages back at `/testbed`.
Across Lite that holds for 294 of 300 instances, and for those the right amount
of preparation is none: replaying `pip install -e .` would rebuild the compiled
repos and cost minutes per instance for no change.

The exceptions matter though. `psf/requests` (6 instances) is installed with a
plain `pip install .`, which copies the tree into site-packages — the agent would
edit `/testbed` and then test the code it started with, all run long.

So `auto` runs, per instance, only what the mount actually breaks: it re-runs a
non-editable install as an editable one, and runs cheap non-pip setup such as
Django's `locale-gen`. `always` replays the whole install step verbatim; `never`
skips it. The commands come from the instance's own `eval_script` in the dataset,
not from a per-repo table here.

One known gap in `auto`: Django's eval script also exports `LC_ALL=en_US.UTF-8`,
and an `export` cannot outlive the shell that ran it, so the agent's own test runs
use the `C.UTF-8` the container sets. That affects what the agent observes, never
how it is graded.

## Why it is shaped like this

**One driver, N shims.** `run_agent.py` owns the workspace, the container, the
objective text, the preparation step and the patch extraction, and `--framework`
changes only which shim it spawns. Separate drivers would drift, and then a
difference in the score would be a difference between the drivers rather than
between the agents.

**No agent is imported.** The byLLM one cannot be: `jac` is a self-contained
binary carrying its own Python, and `jaclang.byllm` exists only inside it, so the
conda Python running the driver cannot import `orchestrator.jac`. The Python ones
are held to the same shape anyway — they keep their repository binding and token
counter in module globals, so N instances in N threads of one process would
interleave into each other's state. All three are therefore one process per
instance: job file in, result file out. stdout is not a channel; byLLM, litellm,
LangChain and httpx all log to it.

**The workspace is the image's, on the host.** Each instance's `/testbed` is
copied out of its SWE-bench image, so the tree the agent gets is at `base_commit`
and already carries the compiled extensions and `.egg-info` that the image's
conda env is installed against. That host copy is bind-mounted back into a
running container at `/testbed`.

**Reads and writes are host-side; commands are container-side.** The agent's file
tools work on the host copy directly. Its `run_command` goes through `docker exec`
into the container (`CODEAGENT_EXEC=docker`), so `pytest` is the instance's pytest
with the instance's dependencies. Without this the Verifying phase would only ever
observe import errors, and the benchmark would be measuring the harness.

**Grading is the harness's verdict, not ours.** `grade.py` builds a fresh
container from the instance image and applies only `model_patch`, so nothing the
agent did to its workspace during inference can reach the score. It does not
reimplement the answer either: `make_test_spec` and `get_eval_report` come from
the installed swebench package, so it parses with the instance's own log parser
and applies the same FAIL_TO_PASS / PASS_TO_PASS rule. What it owns is the
container mechanics and the log capture.

That was checked against known answers rather than assumed:

| fed to the grader | verdict |
|---|---|
| the dataset's own gold patch | resolved |
| a patch that applies cleanly and fixes nothing | unresolved |
| an empty patch | empty, nothing run |

## The patch is not `git diff`

Several SWE-bench images ship a `/testbed` that is already dirty: compiled
extensions, generated headers, `.egg-info`. Diffing against `base_commit` would
sweep all of it into the prediction and the harness would reject the patch. So
`extract_patch` snapshots `git status` at handover, stages only the paths whose
status changed since, and drops paths that are never part of a fix — caches,
bytecode, build output (`is_noise` in `workspace.py`). Test-file edits are left
in: the harness resets the graded test files itself before running.

## Output

`results/<run-id>/`, and `compare.py` gives each side its own,
`results/<run-id>-<framework>/`.

| file | contents |
|---|---|
| `predictions.jsonl` | `instance_id`, `model_name_or_path`, `model_patch` — the harness's input format |
| `runs.jsonl` | per instance: framework, phases, tool calls, LLM calls, prompt/completion/cached tokens, wall clock, error |
| `logs/<instance_id>/job.json` | the objective the agent was given |
| `logs/<instance_id>/result.json` | what the agent reported, including its full phase ledger |
| `logs/<instance_id>/agent.log` | the shim's transcript |
| `eval_logs/<instance_id>/test_output.txt` | the captured grading log |
| `eval_results.jsonl` | one verdict per line, appended as it lands, so grading resumes |
| `<model>.<run-id>.json` | the harness-shaped report |

Every shim writes the same `result.json` keys, so `runs.jsonl` has one shape on
every side and the reports are directly comparable. `../case_study` turns a set
of these run directories into a study.

## What this changes in the agents

One thing, in each agent's `run_command`: an execution backend. With
`CODEAGENT_EXEC=docker` (or `=udocker`) set, it builds a `docker exec` / `udocker
run` invocation instead of a local `Popen`, and the interpreter (`python`,
`python3`) joins the allowlist — Django's suite runs through `tests/runtests.py`
rather than pytest, and Django is 114 of Lite's 300 instances.

Everything else is untouched, and deliberately so: the same allowlist, the same
denied flags, the same path confinement, the same output clipping run before the
backend is consulted, so where a command runs never widens what may run. The
default is still `local`; with `CODEAGENT_EXEC` unset the agents behave exactly as
they do standalone.

## Tests

```bash
python -m pytest tests -q
```

Covers the parts of the bridge that decide numbers: the framework registry, the
noise filter and patch extraction against a real temporary git repo, the
preparation-script derivation, the status partition, and the comparison table at
two, three and four sides.
