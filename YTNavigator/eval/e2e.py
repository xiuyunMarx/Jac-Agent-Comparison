#!/usr/bin/env python3
"""End-to-end YT-Navigator agent evaluation: dataset -> both agents -> scores.

Runs the complete pipeline from an empty machine to a comparison table:

  1. prereqs   - jac binary, OPENAI_API_KEY, Python deps
  2. database  - use a reachable Postgres (POSTGRES_* env), or start a
                 dockerized pgvector instance (container: ytnav-bench-pg)
  3. dataset   - load the synthetic NeuralBytes channel (datasets/build.py):
                 10 videos, ~67 transcript chunks, real bge-small embeddings
  4. sanity    - retrieval smoke query against the loaded data (no LLM)
  5. byLLM     - run the Jac implementation over datasets/questions.jsonl
  6. LangGraph - run the original implementation (needs its own env; skipped
                 with a note if unavailable)
  7. score     - shared evaluator, side-by-side table + out/report.json

Typical usage:
    python e2e.py                          # everything, both implementations
    python e2e.py --impl byllm             # one side only
    python e2e.py --smoke                  # stages 1-4 only, no LLM, no key
    python e2e.py --judge                  # add LLM-judge answer scoring
    python e2e.py --langgraph-python ~/venvs/ytnav/bin/python

The database credentials chosen (or found) are exported to every child
process, so run.py / build.py / both agents all hit the same database.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent                      # YTNavigator/
LANGGRAPH_DIR = ROOT / "YT-Navigator"
OPENAI_SDK_DIR = ROOT / "openai_sdk"
BYLLM_DIR = ROOT / "byLLM"
DATASETS_DIR = ROOT / "datasets"

DOCKER_CONTAINER = "ytnav-bench-pg"
DOCKER_IMAGE = "pgvector/pgvector:pg16"
DOCKER_DB = {"host": "localhost", "port": "5544", "user": "ytnav", "password": "ytnav_bench", "db": "ytnav_bench"}


def stage(title):
    print(f"\n{'=' * 70}\n== {title}\n{'=' * 70}")


def fail(message):
    sys.exit(f"\nERROR: {message}")


def have(cmd):
    from shutil import which

    return which(cmd) is not None


def try_connect(env):
    """Return True if Postgres answers with the given env settings."""
    try:
        import psycopg2

        conn = psycopg2.connect(
            host=env.get("POSTGRES_HOST", "localhost"),
            port=env.get("POSTGRES_PORT", "5432"),
            user=env.get("POSTGRES_USER", "postgres"),
            password=env.get("POSTGRES_PASSWORD", ""),
            dbname=env.get("POSTGRES_DB", "postgres"),
            connect_timeout=3,
        )
        conn.close()
        return True
    except Exception:
        return False


def load_env_file(path):
    values = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def find_local_pg_bin():
    """Find a Postgres bin dir that also ships the pgvector extension.

    Checks PATH first, then conda envs - `conda create -n ytnav-pg -c
    conda-forge postgresql pgvector` produces one without needing root.
    """
    from shutil import which

    candidates = []
    if which("initdb") and which("pg_ctl"):
        candidates.append(Path(which("initdb")).parent)
    for pattern in ("miniconda3/envs/*/bin", "anaconda3/envs/*/bin", "mambaforge/envs/*/bin"):
        candidates.extend(sorted(Path.home().glob(pattern)))
    for bin_dir in candidates:
        if not ((bin_dir / "initdb").exists() and (bin_dir / "pg_ctl").exists()):
            continue
        prefix = bin_dir.parent
        for control in ("share/extension/vector.control", "share/postgresql/extension/vector.control"):
            if (prefix / control).exists():
                return bin_dir
    return None


def start_local_postgres(bin_dir, db_env):
    """Init (once) and start a local Postgres under eval/pgdata; create the DB."""
    data_dir = EVAL_DIR / "pgdata"
    socket_dir = Path("/tmp/ytpg")  # short path: Unix sockets cap at ~107 bytes
    socket_dir.mkdir(exist_ok=True)
    log_file = EVAL_DIR / "pgdata.log"

    if not (data_dir / "PG_VERSION").exists():
        print(f"Initializing local Postgres data dir {data_dir} ...")
        run = subprocess.run(
            [str(bin_dir / "initdb"), "-D", str(data_dir), "-U", db_env["POSTGRES_USER"], "--auth=trust"],
            capture_output=True, text=True,
        )
        if run.returncode != 0:
            fail(f"initdb failed: {run.stderr.strip()[-500:]}")

    status = subprocess.run([str(bin_dir / "pg_ctl"), "-D", str(data_dir), "status"], capture_output=True)
    if status.returncode != 0:  # not running
        print(f"Starting local Postgres (port {db_env['POSTGRES_PORT']}, data {data_dir}) ...")
        run = subprocess.run(
            [str(bin_dir / "pg_ctl"), "-D", str(data_dir), "-l", str(log_file),
             "-o", f"-p {db_env['POSTGRES_PORT']} -k {socket_dir}", "start"],
            capture_output=True, text=True,
        )
        if run.returncode != 0:
            fail(f"pg_ctl start failed (see {log_file}): {run.stderr.strip()[-500:]}")

    subprocess.run(
        [str(bin_dir / "createdb"), "-h", db_env["POSTGRES_HOST"], "-p", db_env["POSTGRES_PORT"],
         "-U", db_env["POSTGRES_USER"], db_env["POSTGRES_DB"]],
        capture_output=True,  # fails harmlessly if the DB already exists
    )
    print(f"(Local Postgres keeps running for future runs; stop with: "
          f"{bin_dir}/pg_ctl -D {data_dir} stop)")


def ensure_database(args):
    """Make a Postgres with pgvector reachable; export POSTGRES_* into os.environ."""
    stage("Stage 2/7: database")

    # Priority 1: explicitly configured / .env-provided database.
    file_env = load_env_file(LANGGRAPH_DIR / ".env")
    for key, value in file_env.items():
        os.environ.setdefault(key, value)
    if os.environ.get("POSTGRES_HOST") and try_connect(os.environ):
        print(f"Using configured Postgres at {os.environ['POSTGRES_HOST']}:"
              f"{os.environ.get('POSTGRES_PORT', '5432')}")
        return

    # Priority 2: a benchmark Postgres started earlier (docker or local) on 5544.
    docker_env = {
        "POSTGRES_HOST": DOCKER_DB["host"], "POSTGRES_PORT": DOCKER_DB["port"],
        "POSTGRES_USER": DOCKER_DB["user"], "POSTGRES_PASSWORD": DOCKER_DB["password"],
        "POSTGRES_DB": DOCKER_DB["db"],
    }
    if try_connect(docker_env):
        os.environ.update(docker_env)
        print(f"Using existing benchmark Postgres on port {DOCKER_DB['port']}")
        return

    docker_ok = False
    if have("docker"):
        docker_ok = subprocess.run(["docker", "ps"], capture_output=True).returncode == 0

    # Priority 3: no usable docker -> manage a local (conda) Postgres ourselves.
    if not docker_ok:
        bin_dir = find_local_pg_bin()
        if bin_dir is not None:
            print(f"Docker unavailable - using local Postgres binaries from {bin_dir}")
            start_local_postgres(bin_dir, docker_env)
            for _ in range(30):
                if try_connect(docker_env):
                    os.environ.update(docker_env)
                    print("Database is up.")
                    return
                time.sleep(1)
            fail(f"local Postgres did not become ready (see {EVAL_DIR / 'pgdata.log'})")
        fail("No reachable Postgres, docker is unusable, and no local Postgres binaries found.\n"
             "Fix one of:\n"
             "  - conda create -y -n ytnav-pg -c conda-forge postgresql pgvector   (then rerun; no root needed)\n"
             "  - sudo usermod -aG docker $USER   (then re-login and rerun)\n"
             f"  - sudo docker run -d --name {DOCKER_CONTAINER} -p {DOCKER_DB['port']}:5432 "
             f"-e POSTGRES_USER={DOCKER_DB['user']} -e POSTGRES_PASSWORD={DOCKER_DB['password']} "
             f"-e POSTGRES_DB={DOCKER_DB['db']} {DOCKER_IMAGE}   (then rerun)\n"
             "  - point POSTGRES_* at any Postgres that has the pgvector extension")

    # Priority 4: docker works - start the containerized instance.
    print(f"Starting {DOCKER_IMAGE} as container '{DOCKER_CONTAINER}' on port {DOCKER_DB['port']} ...")
    subprocess.run(["docker", "rm", "-f", DOCKER_CONTAINER], capture_output=True)
    run = subprocess.run(
        ["docker", "run", "-d", "--name", DOCKER_CONTAINER,
         "-p", f"{DOCKER_DB['port']}:5432",
         "-e", f"POSTGRES_USER={DOCKER_DB['user']}",
         "-e", f"POSTGRES_PASSWORD={DOCKER_DB['password']}",
         "-e", f"POSTGRES_DB={DOCKER_DB['db']}",
         DOCKER_IMAGE],
        capture_output=True, text=True,
    )
    if run.returncode != 0:
        fail(f"docker run failed: {run.stderr.strip()}")

    for _ in range(60):
        if try_connect(docker_env):
            os.environ.update(docker_env)
            print("Database is up.")
            print(f"(Container '{DOCKER_CONTAINER}' keeps running for future runs; "
                  f"remove with: docker rm -f {DOCKER_CONTAINER})")
            return
        time.sleep(1)
    fail("Postgres container did not become ready within 60s")


def ensure_django_schema(args):
    """Bring the database under Django's migrations without losing the dataset.

    datasets/build.py creates the data tables directly so byLLM-only runs need
    no Django. But Django's initial migration also creates app_user (the
    custom auth model), so `migrate --fake-initial` refuses to fake it while
    app_user is missing and then trips over the existing app_channel. In that
    state: drop the builder-created tables, let Django create the canonical
    schema, and rebuild the dataset into it.
    """
    import psycopg2

    conn = psycopg2.connect(
        host=os.environ["POSTGRES_HOST"], port=os.environ["POSTGRES_PORT"],
        user=os.environ["POSTGRES_USER"], password=os.environ.get("POSTGRES_PASSWORD", ""),
        dbname=os.environ["POSTGRES_DB"],
    )
    with conn, conn.cursor() as cur:
        cur.execute("SELECT to_regclass('app_user'), to_regclass('app_channel')")
        has_user, has_channel = [r is not None for r in cur.fetchone()]

    run_step([args.langgraph_python, "manage.py", "makemigrations", "app"],
             cwd=LANGGRAPH_DIR, name="makemigrations")

    if has_channel and not has_user:
        print("Dataset tables predate Django - recreating them under Django's schema "
              "and rebuilding the dataset ...")
        with conn, conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS app_videochunk, app_video, app_channel CASCADE")
        conn.close()
        run_step([args.langgraph_python, "manage.py", "migrate"], cwd=LANGGRAPH_DIR, name="migrate")
        build_cmd = [sys.executable, str(DATASETS_DIR / "build.py"), "--replace"]
        if args.fake_embeddings:
            build_cmd.append("--fake-embeddings")
        run_step(build_cmd, name="dataset rebuild")
    else:
        conn.close()
        run_step([args.langgraph_python, "manage.py", "migrate", "--fake-initial"],
                 cwd=LANGGRAPH_DIR, name="migrate")


def run_step(cmd, cwd=None, name=None):
    name = name or " ".join(str(c) for c in cmd)
    print(f"$ {' '.join(str(c) for c in cmd)}")
    proc = subprocess.run([str(c) for c in cmd], cwd=cwd)
    if proc.returncode != 0:
        fail(f"step failed: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--impl", choices=["all", "both", "byllm", "langgraph", "openai_sdk"],
                        default="all",
                        help="'both' = byllm + langgraph (the historical pair); "
                             "'all' adds the no-framework openai_sdk baseline")
    parser.add_argument("--questions", default=str(DATASETS_DIR / "questions.jsonl"))
    parser.add_argument("--langgraph-python", default=sys.executable,
                        help="Interpreter with YT-Navigator's deps (default: this one)")
    parser.add_argument("--replace-data", action="store_true", help="Rebuild the dataset even if already loaded")
    parser.add_argument("--fake-embeddings", action="store_true",
                        help="Pipeline smoke without torch - retrieval quality meaningless")
    parser.add_argument("--smoke", action="store_true", help="Stop after the retrieval sanity check (no LLM calls)")
    parser.add_argument("--judge", action="store_true", help="Add LLM-as-judge scoring")
    parser.add_argument("--judge-model", default=os.environ.get("EVAL_JUDGE_MODEL", ""),
                        help="litellm model name for the judge "
                             "(default: $EVAL_JUDGE_MODEL, else evaluate.py's own)")
    args = parser.parse_args()

    # ---- Stage 1: prerequisites -------------------------------------------
    stage("Stage 1/7: prerequisites")
    problems = []
    if args.impl in ("all", "both", "byllm") and not have("jac"):
        problems.append("jac binary not on PATH (needed for the byLLM implementation)")
    try:
        import psycopg2  # noqa: F401
    except ImportError:
        problems.append("psycopg2 missing: pip install psycopg2-binary")
    if not args.fake_embeddings:
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            problems.append("sentence-transformers missing (needed for real embeddings): "
                            "pip install sentence-transformers   (or use --fake-embeddings for a smoke run)")
    if not args.smoke:
        key = os.environ.get("OPENAI_API_KEY") or load_env_file(LANGGRAPH_DIR / ".env").get("OPENAI_API_KEY")
        if not key:
            problems.append("OPENAI_API_KEY not set (env or YT-Navigator/.env) - required for agent runs "
                            "(use --smoke to validate everything up to the LLM calls without a key)")
        else:
            os.environ["OPENAI_API_KEY"] = key
    if problems:
        fail("prerequisites missing:\n  - " + "\n  - ".join(problems))
    if args.fake_embeddings:
        # Query-side retrieval (stage 4 and the byLLM agent) must use the same
        # fake vectors the dataset was built with.
        os.environ["YTNAV_FAKE_EMBEDDINGS"] = "1"
    print("All prerequisites present.")

    # ---- Stage 2: database ------------------------------------------------
    ensure_database(args)

    # ---- Stage 3: dataset -------------------------------------------------
    stage("Stage 3/7: dataset (synthetic NeuralBytes channel)")
    build_cmd = [sys.executable, str(DATASETS_DIR / "build.py")]
    if args.replace_data:
        build_cmd.append("--replace")
    if args.fake_embeddings:
        build_cmd.append("--fake-embeddings")
    run_step(build_cmd, name="dataset build")

    # ---- Stage 4: retrieval sanity ----------------------------------------
    stage("Stage 4/7: retrieval sanity check (no LLM)")
    sys.path.insert(0, str(BYLLM_DIR))
    import retrieval  # noqa: E402

    channel_id = retrieval.resolve_channel(os.environ.get("YTNAV_CHANNEL", ""))
    print(f"Channel resolved: {channel_id}")
    result = retrieval.search_videos_impl("how does self attention use queries keys and values", channel_id)
    count_result = retrieval.run_sql_impl("SELECT COUNT(*) AS n FROM app_video")
    print(f"SQL tool: video count -> {count_result}")
    if args.fake_embeddings:
        print("Semantic search executed (results meaningless with fake embeddings).")
    elif "vTRANSF0001" in result:
        print("Semantic search OK: transformer video ranked for a self-attention query.")
    else:
        fail("Semantic search sanity failed: expected vTRANSF0001 in results for a self-attention query.\n"
             f"Got:\n{result[:800]}")
    if args.smoke:
        print("\nSmoke run complete - dataset loaded and retrieval verified. "
              "Rerun without --smoke (and with OPENAI_API_KEY) for the full eval.")
        return

    # ---- Stage 5+6: implementations ---------------------------------------
    results = []
    if args.impl in ("all", "both", "byllm"):
        stage("Stage 5/7: byLLM implementation")
        run_step([sys.executable, str(EVAL_DIR / "run.py"), "--impl", "byllm",
                  "--questions", args.questions, "--no-score"], name="byLLM run")
        results.append(EVAL_DIR / "out" / "results_byllm.jsonl")

    if args.impl in ("all", "openai_sdk"):
        stage("Stage 5b/7: openai_sdk implementation")
        # Runs in this interpreter, which already needed psycopg2 for the
        # retrieval check above; the only extra is the openai package.
        probe = subprocess.run([sys.executable, "-c", "import openai"],
                               capture_output=True, text=True)
        if probe.returncode != 0:
            print(f"SKIPPED: `openai` is not importable with {sys.executable}. "
                  f"Install it (pip install -e {OPENAI_SDK_DIR}) and rerun with --impl openai_sdk.")
        else:
            run_step([sys.executable, str(EVAL_DIR / "run.py"), "--impl", "openai_sdk",
                      "--questions", args.questions, "--no-score"], name="openai_sdk run")
            results.append(EVAL_DIR / "out" / "results_openai_sdk.jsonl")

    if args.impl in ("all", "both", "langgraph"):
        stage("Stage 6/7: LangGraph implementation")
        probe = subprocess.run(
            [args.langgraph_python, "-c", "import django, langgraph, langchain_openai"],
            capture_output=True, text=True,
        )
        if probe.returncode != 0:
            print("SKIPPED: the LangGraph implementation's dependencies are not available in "
                  f"{args.langgraph_python}.\nInstall them (pip install -e {LANGGRAPH_DIR}) and rerun with "
                  "--impl langgraph --langgraph-python <that interpreter>.")
        else:
            # First run against a fresh database needs Django's tables.
            ensure_django_schema(args)
            run_step([sys.executable, str(EVAL_DIR / "run.py"), "--impl", "langgraph",
                      "--questions", args.questions, "--no-score",
                      "--langgraph-python", args.langgraph_python], name="LangGraph run")
            results.append(EVAL_DIR / "out" / "results_langgraph.jsonl")

    # ---- Stage 7: scoring -------------------------------------------------
    stage("Stage 7/7: scoring")
    existing = [r for r in results if r.is_file()]
    if not existing:
        fail("no result files produced")
    score_cmd = [sys.executable, str(EVAL_DIR / "score.py")] + [str(r) for r in existing] + [
        "--questions", args.questions, "--report", str(EVAL_DIR / "out" / "report.json")]
    if args.judge:
        score_cmd.append("--judge")
        if args.judge_model:
            score_cmd += ["--judge-model", args.judge_model]
    run_step(score_cmd, name="scoring")
    print(f"\nDone. Result files: {', '.join(str(r) for r in existing)}")
    print(f"Full report: {EVAL_DIR / 'out' / 'report.json'}")


if __name__ == "__main__":
    main()
