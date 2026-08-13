# Coding agents ⟶ SWE-bench

Runs either coding agent over SWE-bench instances and grades the result with the
official harness in `../SWE-bench`. The two agents are the two halves of a
framework comparison:

| framework | agent | shim |
|---|---|---|
| `byllm` | `../byLLM` — Jac phase graph, byLLM | `swe_entry.jac` |
| `langgraph` | `../langgraph` — LangGraph state machine | `swe_entry.py` |

```
run_agent.py     inference: instances -> predictions.jsonl (+ runs.jsonl)
swe_entry.jac    the Jac side of one instance
swe_entry.py     the LangGraph side of one instance
evaluate.py      grading: predictions.jsonl -> resolve rate, joined with agent cost
grade_local.py   the same grading without a Docker daemon
compare.py       both agents over the same instances, graded, side by side
```

## No Docker group? Use `--runtime udocker`

Everything here works without Docker, without root and without being in the
`docker` group:

```bash
pip install udocker && udocker install
python compare.py --runtime udocker --repo psf/requests --limit 2 --run-id smoke
```

udocker pulls the instance images itself and unpacks them into plain
directories, running commands under PRoot -- which fakes uid 0 by intercepting
syscalls rather than by holding any privilege. It is not a second-class path:
the same images, the same `/testbed`, the same conda env, the same tests.

It is in one respect *simpler* than the daemon. Under Docker the image's
`/testbed` has to be copied out with `docker cp`, bind-mounted back over itself,
and then chowned afterwards because everything the container wrote landed
root-owned. Under udocker the container's `/testbed` already **is** a host
directory owned by you, so all three steps disappear and there is only ever one
copy of the tree.

Grading has no daemon-free path in the official harness -- `run_evaluation`
speaks to the Docker API directly -- so `grade_local.py` drives the same
evaluation through the runtime interface instead. It does not reimplement the
verdict: `make_test_spec` and `get_eval_report` come from the installed swebench
package, so it parses with the instance's own log parser and applies the same
FAIL_TO_PASS/PASS_TO_PASS rule. What it owns is only the container mechanics and
the log capture.

That was checked against known answers rather than assumed:

| fed to the grader | verdict |
|---|---|
| the dataset's own gold patch | resolved |
| a patch that applies cleanly and fixes nothing | unresolved |
| an empty patch | empty, nothing run |

## Why it is shaped like this

**One driver, two shims.** `run_agent.py` owns the workspace, the container, the
objective text, the preparation step and the patch extraction, and `--framework`
changes only which shim it spawns. Two drivers would drift, and then a
difference in the score would be a difference between the drivers rather than
between the agents.

**Neither agent is imported.** The byLLM one cannot be: `jac` is a self-contained
binary carrying its own Python 3.14, and `jaclang.byllm` exists only inside it,
so the conda Python running the driver cannot import `orchestrator.jac`. The
LangGraph one is held to the same shape anyway — it keeps its repository binding
and token counter in module globals, so N instances in N threads of one process
would interleave into each other's state. Both are therefore one process per
instance: job file in, result file out. stdout is not a channel; byLLM, litellm,
LangChain and httpx all log to it.

**The workspace is the image's, on the host.** Each instance's `/testbed` is
copied out of its SWE-bench image with `docker cp`, so the tree the agent gets
is at `base_commit` and already carries the compiled extensions and `.egg-info`
that the image's conda env is installed against. That host copy is then
bind-mounted back into a running container at `/testbed`.

**Reads and writes are host-side; commands are container-side.** The agent's
file tools work on the host copy directly. Its `run_command` goes through
`docker exec` into the container (`CODEAGENT_EXEC=docker`), so `pytest` is the
instance's pytest with the instance's dependencies. Without this the Verifying
phase would only ever observe import errors, and the benchmark would be
measuring the harness rather than the agent.

**Grading is untouched.** `evaluate.py` shells out to
`swebench.harness.run_evaluation` unmodified. It builds its own fresh container
from the instance image and applies only `model_patch`, so nothing the agent did
to its workspace during inference can reach the score.

## Setup

1. **A container runtime.** Either the Docker daemon reachable as your own user:

   ```bash
   sudo usermod -aG docker $USER    # then open a new login shell
   docker ps                        # must succeed with no sudo
   ```

   or, with no sudo and no group membership, udocker:

   ```bash
   pip install udocker && udocker install
   ```

   and pass `--runtime udocker` to everything below.

2. **The harness package** (already installed if you ran this once):

   ```bash
   pip install -e ../SWE-bench
   ```

3. **A provider key** for the model the agents call:

   ```bash
   export OPENAI_API_KEY=sk-...
   ```

4. **Both agents loadable by their own runners.** Neither project needs
   installing — each shim puts its agent home on `sys.path` itself — but the
   runners do need to exist: `jac` on PATH for the byLLM side, and for the
   LangGraph side the interpreter given by `--python` (default: the one running
   the driver) needs that project's dependencies importable:

   ```bash
   pip install langgraph langchain-core langchain-openai pydantic
   python -c "import langgraph, langchain_openai"   # must be silent
   ```

   `run_agent.py` checks this for you: it loads the agent once at startup —
   before pulling any image or spawning any instance — and warns if it did not
   come up cleanly.

5. **Disk.** Each instance image is 1–2 GB and each workspace is a full repo
   copy. Workspaces are deleted after their patch is taken (`--keep-workspaces`
   to keep them); images are not, unless you pass `--cleanup-images`. A full
   Lite run touches ~12 repos' worth of images, and an A/B run does the
   workspaces twice.

## Run both

This is the point of the directory. Smoke-test first — `psf/requests` is the
cheapest repo in Lite:

```bash
python compare.py --repo psf/requests --limit 2 --run-id smoke --workers 2
```

Then the full set:

```bash
python compare.py --run-id lite-01 --workers 8 --eval-workers 8
```

`compare.py` runs inference for each framework, grades each, and prints the A/B.
Two things it does that running the four commands by hand does not:

* **The instance set is resolved once and pinned.** Both sides are handed the
  same explicit `--instance-ids`, so `--limit 20` cannot mean a different twenty
  on the second side. An A/B over two different instance sets is not an A/B.
* **The frameworks run one after another, never at once.** They would otherwise
  contend for the same cores, disk and docker daemon, and the wall-clock and
  token numbers are meant to be compared against each other.

Any flag it does not recognise is forwarded to `run_agent.py`, so `--workers 8`,
`--max-steps 12`, `--network none` and the rest work here too. The flags that
must be identical on both sides — `--dataset`, `--split`, `--repo`, `--limit`,
`--instance-ids`, `--run-id`, `--output-dir` — belong to `compare.py` itself and
are refused in the passthrough.

## Run one

```bash
python run_agent.py --framework langgraph --repo psf/requests --limit 2 --run-id smoke-lg
python evaluate.py --predictions results/smoke-lg/predictions.jsonl
```

`run_agent.py` is resumable: it skips instances already in `predictions.jsonl`,
so an interrupted run continues by re-issuing the same command with the same
`--run-id`. `--force` re-runs them instead. `compare.py` inherits this — the same
`--run-id` resumes both sides.

Already have two graded runs? Put them side by side without re-grading:

```bash
python evaluate.py --compare results/lite-01-byllm results/lite-01-langgraph
```

### Options worth knowing

| flag | why |
|---|---|
| `--framework` | `byllm` or `langgraph`. Picks the agent and, with it, the shim, the agent home, the container prefix and the `model_name_or_path`. |
| `--runtime` | `docker` or `udocker`. Picks how the image is obtained and where commands run, for inference *and* grading. |
| `--udocker-dir` | udocker's image and container repository (default `~/.udocker`). Images are 1-2 GB each; point this at a disk with room. |
| `--model` | the model the agent calls (default `gpt-4o`). Pin it identically across the frameworks, or you are measuring the model. |
| `--max-steps` | phase budget for one agent run (default 10). The agents' own loop brakes are per-phase; this bounds the traversal. |
| `--workers` | instances in flight. Each holds one container and one agent process. |
| `--instance-timeout` | wall-clock ceiling per instance (default 1800s). |
| `--limit` / `--repo` / `--instance-ids` | subset selection, applied in that order. |
| `--network none` | cut the workspace container off the network. Some suites fail without it; most do not need it. |
| `--prepare` | `auto` (default) / `always` / `never` — see below. |
| `--jac` / `--python` | the runner for each side. Only the selected framework's is used. |
| `--exec-backend` | where the agent's own commands run. Defaults to `--runtime`, so it is normally not worth setting; `local` runs them on this machine, which for these repos means unusable test feedback. |

### `--prepare`

Bind-mounting the workspace over `/testbed` is transparent only when the image
installed the repo *editably*, which points site-packages back at `/testbed`.
Across Lite that holds for 294 of 300 instances, and for those the right amount
of preparation is none: replaying `pip install -e .` would rebuild the compiled
repos and cost minutes per instance for no change.

The exceptions matter though. `psf/requests` (6 instances) is installed with a
plain `pip install .`, which copies the tree into site-packages — the agent
would edit `/testbed` and then test the code it started with, all run long.

So `auto` runs, per instance, only what the mount actually breaks: it re-runs a
non-editable install as an editable one, and runs cheap non-pip setup such as
Django's `locale-gen`. `always` replays the whole install step verbatim; `never`
skips it. The commands come from the instance's own `eval_script` in the
dataset, not from a per-repo table here.

One known gap in `auto`: Django's eval script also exports `LC_ALL=en_US.UTF-8`,
and an `export` cannot outlive the shell that ran it, so the agent's own test
runs use the `C.UTF-8` that `container_env` sets. That affects what the agent
observes, never how it is graded.

## Output

`results/<run-id>/`, and `compare.py` gives each side its own:
`results/<run-id>-byllm/` and `results/<run-id>-langgraph/`.

| file | contents |
|---|---|
| `predictions.jsonl` | `instance_id`, `model_name_or_path`, `model_patch` — the harness's input format |
| `runs.jsonl` | per instance: phases, tool calls, LLM calls, prompt/completion tokens, wall clock, error |
| `logs/<instance_id>/job.json` | the objective the agent was given |
| `logs/<instance_id>/result.json` | what the agent reported, including its full phase ledger |
| `logs/<instance_id>/agent.log` | the shim's transcript |
| `<model>.<run-id>.json` | the harness's report, written here by `evaluate.py` |

Both shims write the same `result.json` keys, so `runs.jsonl` has one shape on
both sides and the reports are directly comparable.

`evaluate.py` prints the resolve rate joined against `runs.jsonl`, so the number
arrives next to what it cost — tokens per resolved instance, and how resolved and
unresolved runs differ in tool calls and phases. `--compare` adds the part that
only an A/B has: which instances *both* frameworks resolved, and which only one
did. Two resolve rates alone do not say whether the agents solved the same
problems, and the disagreement set is what you actually go and open.

```
  Agreement over 6 instance(s) run by both
    resolved by both         : 2
    only smoke-byllm         : 1
    only smoke-langgraph     : 2
    resolved by neither      : 1
```

## The patch is not `git diff`

Several SWE-bench images ship a `/testbed` that is already dirty: compiled
extensions, generated headers, `.egg-info`. Diffing against `base_commit` would
sweep all of it into the prediction and the harness would reject the patch. So
`extract_patch` snapshots `git status` at handover, stages only the paths whose
status changed since, and drops paths that are never part of a fix — caches,
bytecode, build output (`is_noise` in `run_agent.py`). Test-file edits are left
in: the harness resets the graded test files itself before running.

## What this changes in the agents

One thing, in each agent's `run_command`: an execution backend. With
`CODEAGENT_EXEC=docker` (or `=udocker`) set, it builds a `docker exec` / `udocker
run` invocation instead of a local `Popen`, and the interpreter (`python`,
`python3`) joins the allowlist —
Django's suite runs through `tests/runtests.py` rather than pytest, and Django is
114 of Lite's 300 instances.

Everything else is untouched, and deliberately so: the same allowlist, the same
denied flags, the same path confinement, the same output clipping run before the
backend is consulted, so where a command runs never widens what may run. The
default is still `local`; with `CODEAGENT_EXEC` unset both agents behave exactly
as they did. The two container backends differ only in how the command is
spawned — the allowlist they widen and the screening they run are identical, and
the tests assert that for both. `../byLLM/tests/verify_tests.jac` and
`../langgraph/tests/test_verify.py` pin both halves of that on their respective
sides.

The two backends were checked against each other directly — the same command
strings through both `run_command` implementations, compared byte for byte —
so the action space really is the same one. See `../langgraph/README.md` for
that comparison and the one discrepancy it found.
