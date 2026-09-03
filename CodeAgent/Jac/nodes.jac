"""阶段节点 + 工具箱。每个节点 = 一个目标(goal) + 一组它被允许使用的工具。"""
import os;
import re;
import shlex;
import subprocess;
import warnings;
import from jaclang.byllm.lib { Model, IterationAction, IterationContext }

# GLM 5.2 走 ollama 云端。注意用 OpenAI 兼容端点（openai/ + https://ollama.com/v1）而不是
# litellm 的 ollama_chat/ 提供者：后者不发 Authorization 头（云端必 401），还会把 tools
# 降级成 "JSON only" prompt，byLLM 的原生 tool-call 就没了。三个环境变量都可覆盖。
#
# byLLM 0.31.1 在 litellm 路径上会把 config 里的 base_url pop 掉而不传给 litellm
# （basellm.impl.jac dispatch_no_streaming），所以这里通过 litellm 自己认的
# OPENAI_API_BASE 环境变量把端点指过去；已显式设置时不覆盖。
glob MODEL_NAME: str = os.environ.get("CODEAGENT_MODEL", "openai/glm-5.2");
glob MODEL_BASE: str = os.environ.get("OLLAMA_API_BASE", "https://ollama.com/v1");
glob _base_set: bool = (
    (not MODEL_NAME.startswith("openai/"))
    or (os.environ.setdefault("OPENAI_API_BASE", MODEL_BASE) is not None)
);
glob _quiet: bool = (warnings.filterwarnings("ignore", message="Pydantic serializer warnings") is None);
glob llm = Model(
    model_name=MODEL_NAME,
    api_key=os.environ.get("OLLAMA_API_KEY", ""),
    config={"base_url": MODEL_BASE}
);

# ---------------------------------------------------------------------------
# 整个 run 共用的状态。关键设计：一段对话跑完整个 run，阶段切换只换工具和目标。
# ---------------------------------------------------------------------------
glob history: list[dict] = [];          # byLLM 会原地 append 每一轮
glob tool_log: list[dict] = [];         # 每次工具调用一条 {"name": .., "ok": ..}
glob TOOL_BUDGET: int = 60;             # 预算按工具调用数算，不按阶段数算
glob SCRATCH: str = ".agent_scratch";   # 复现脚本放这里，Finish 阶段删掉
glob WRITE_TOOLS: list[str] = ["replace_in_file", "write_file"];

def log_call(name: str, ok: bool = True, args: str = "") -> dict {
    call = {"name": name, "ok": ok, "args": args};
    tool_log.append(call);
    return call;
}

"""是否真的落了一次成功的写操作 —— 问工具日志，不问模型的总结。"""
def written() -> bool {
    for c in tool_log {
        if c["name"] in WRITE_TOOLS and c["ok"] { return True; }
    }
    return False;
}

"""最后一次写之后有没有跑过命令。没有的话，之前看到的所有测试输出都是旧树的。"""
def ran_since_edit() -> bool {
    ran = False;
    for c in tool_log {
        if (c["name"] in WRITE_TOOLS and c["ok"]) or c["name"] == "revert_file" { ran = False; }
        elif c["name"] in ["run_command", "run_snippet"] { ran = True; }
    }
    return ran;
}

"""ReAct 循环的刹车：只看总预算。不要在这里惩罚重复读文件 —— 上一版就是这么死的。"""
def guard(ctx: IterationContext) -> IterationAction {
    if len(tool_log) >= TOOL_BUDGET { return IterationAction.ABORT_WITH_SUMMARY; }
    return IterationAction.CONTINUE;
}

# ---------------------------------------------------------------------------
# 工具箱。所有工具返回 str，永不抛异常、永不返回空串：
#   成功读取 -> 内容；成功修改 -> "OK: ..."；可恢复错误 -> "Error: ... 下一步该做什么"
# ---------------------------------------------------------------------------
obj Repo {
    has base_dir: str = ".";

    def _abs(path: str) -> str | None {
        base = os.path.realpath(self.base_dir);
        cand = os.path.realpath(os.path.join(base, path));
        if cand == base or cand.startswith(base + os.sep) { return cand; }
        return None;
    }

    def _is_test_path(path: str) -> bool {
        parts = path.replace("\\", "/").split("/");
        name = parts[-1];
        return ("tests" in parts) or ("test" in parts) or name.startswith("test_") or name.endswith("_test.py");
    }

    def read_file(path: str, start_line: int = 1, end_line: int = 0) -> str {
        log_call("read_file", args=f"{path}:{start_line}-{end_line}");
        target = self._abs(path);
        if target is None or not os.path.isfile(target) {
            return f"Error: no such file '{path}'. Use grep to locate it.";
        }
        try {
            with open(target, "r", encoding="utf-8", errors="strict", newline="") as f {
                lines = f.read().splitlines();
            }
        } except UnicodeDecodeError { return f"Error: '{path}' is not UTF-8 text."; }
        total = len(lines);
        first = max(1, start_line);
        last = total if end_line <= 0 else min(total, end_line);
        if total <= 400 { first = 1; last = total; }          # 小文件整份给，避免来回开窗
        body = "\n".join(lines[first - 1:last]);
        return f"# {path} lines {first}-{last} of {total}\n{body}";
    }

    def grep(pattern: str, path: str = ".", file_glob: str = "*.py") -> str {
        log_call("grep", args=f"{pattern} in {path} ({file_glob})");
        try { rx = re.compile(pattern); }
        except re.error as e { return f"Error: bad regex '{pattern}': {str(e)}"; }
        base = self._abs(path);
        if base is None { return f"Error: '{path}' is outside the repository."; }
        import from fnmatch { fnmatch }
        hits: list[str] = [];
        for (d, dirs, files) in os.walk(base) {
            dirs[:] = sorted([x for x in dirs if not x.startswith(".") and x not in ["build", "dist", "__pycache__", "node_modules"]]);
            for name in sorted(files) {
                if not fnmatch(name, file_glob) { continue; }
                full = os.path.join(d, name);
                try {
                    with open(full, "r", encoding="utf-8", errors="ignore") as f { text = f.read(); }
                } except OSError { continue; }
                rel = os.path.relpath(full, os.path.realpath(self.base_dir));
                for (i, line) in enumerate(text.splitlines()) {
                    if rx.search(line) { hits.append(f"{rel}:{i + 1}: {line.strip()[:200]}"); }
                    if len(hits) >= 100 { return "\n".join(hits) + "\n... [100 matches; narrow the pattern]"; }
                }
            }
        }
        return "\n".join(hits) if hits else f"No matches for '{pattern}' under '{path}'.";
    }

    def outline(path: str) -> str {
        log_call("outline", args=path);
        target = self._abs(path);
        if target is None or not os.path.isfile(target) { return f"Error: no such file '{path}'."; }
        with open(target, "r", encoding="utf-8", errors="ignore") as f { lines = f.read().splitlines(); }
        out: list[str] = [];
        for (i, line) in enumerate(lines) {
            if re.`match(r"^\s*(async\s+def|def|class)\s+\w+", line) { out.append(f"{i + 1}: {line.rstrip()}"); }
        }
        return "\n".join(out) if out else f"({path}: no def/class found)";
    }

    def replace_in_file(path: str, old: str, new: str) -> str {
        call = log_call("replace_in_file", ok=False, args=path);
        target = self._abs(path);
        if target is None or not os.path.isfile(target) { return f"Error: no such file '{path}'."; }
        if self._is_test_path(path) {
            return "BLOCKED: test files are reset before grading; edit the library source instead.";
        }
        if not old { return "Error: 'old' must not be empty."; }
        with open(target, "r", encoding="utf-8", newline="") as f { text = f.read(); }
        n = text.count(old);
        if n == 0 {
            hint = " A near-match differs only in whitespace; re-read the file and copy exactly." if old.strip() in text else "";
            return f"Error: 'old' not found in {path}.{hint}";
        }
        if n > 1 { return f"Error: 'old' occurs {n} times in {path}; add surrounding lines to make it unique."; }
        updated = text.replace(old, new);
        with open(target, "w", encoding="utf-8", newline="") as f { f.write(updated); }
        call["ok"] = True;
        warn = "";
        if path.endswith(".py") {
            try { compile(updated, path, "exec"); }
            except SyntaxError as e { warn = f"\nWARNING: {path} no longer parses at line {e.lineno}: {e.msg}. Fix before running tests."; }
        }
        line_no = text[:text.index(old)].count("\n") + 1;
        return f"OK: replaced 1 occurrence in {path} at line {line_no}.{warn}";
    }

    def view_diff() -> str {
        log_call("view_diff");
        return self._exec(["git", "diff", "--no-color", "--", ".", f":(exclude){SCRATCH}"]);
    }

    def run_command(command: str) -> str {
        log_call("run_command", args=command);
        try { argv = shlex.split(command); }
        except ValueError as e { return f"Error: cannot parse command: {str(e)}"; }
        if not argv { return "Error: empty command."; }
        allowed = ["pytest", "python", "python3", "git"];
        if argv[0] not in allowed {
            return f"Error: '{argv[0]}' is not allowed. Allowed: {', '.join(allowed)}.";
        }
        if argv[0] == "git" and (len(argv) < 2 or argv[1] not in ["status", "diff", "log", "show"]) {
            return "Error: only read-only git subcommands (status/diff/log/show) are allowed.";
        }
        if argv[0] == "pytest" { argv = argv + ["-q", "--tb=line", "-p", "no:cacheprovider"]; }
        return self._exec(argv);
    }

    def run_snippet(code: str) -> str {
        log_call("run_snippet", args=code[:80]);
        scratch = os.path.join(os.path.realpath(self.base_dir), SCRATCH);
        os.makedirs(scratch, exist_ok=True);
        script = os.path.join(scratch, "snippet.py");
        with open(script, "w", encoding="utf-8") as f { f.write(code); }
        # python -c 让 sys.path[0] 是 cwd（仓库根），而不是脚本所在的 scratch 目录。
        return self._exec([
            "python", "-c",
            "import runpy, sys; runpy.run_path(sys.argv[1], run_name='__main__')",
            f"{SCRATCH}/snippet.py"
        ]);
    }

    # 容器里先激活镜像的 conda 环境（SWE-bench 镜像的依赖都在 testbed 环境里，
    # 裸 `docker exec ... python` 打到的是系统 python，会缺 pytz/sqlparse 之类），
    # 再 exec 原命令；"$@" 保证 argv 不经过二次 shell 解析。
    def _in_env(argv: list[str]) -> list[str] {
        return ["bash", "-c",
                "source /opt/miniconda3/bin/activate >/dev/null 2>&1; "
                "conda activate testbed >/dev/null 2>&1; exec \"$@\"", "_"] + argv;
    }

    # 唯一的执行点。CODEAGENT_EXEC=docker|udocker 时进容器（依赖只在镜像里有）。
    def _exec(argv: list[str]) -> str {
        backend = os.environ.get("CODEAGENT_EXEC", "local");
        workdir = os.environ.get("CODEAGENT_EXEC_WORKDIR", "/testbed");
        container = os.environ.get("CODEAGENT_EXEC_CONTAINER", "");
        if backend == "docker" {
            prefix = ["docker", "exec", "-w", workdir];
            user = os.environ.get("CODEAGENT_EXEC_USER", "");
            if user {
                prefix += ["-u", user];
            }
            argv = prefix + [container] + self._in_env(argv);
        } elif backend == "udocker" {
            argv = [
                os.environ.get("CODEAGENT_UDOCKER", "udocker"), "run", "--nobanner",
                f"--workdir={workdir}", container
            ] + self._in_env(argv);
        }
        try {
            done = subprocess.run(argv, cwd=self.base_dir, capture_output=True, text=True, timeout=600, stdin=subprocess.DEVNULL);
        } except subprocess.TimeoutExpired { return "Error: command timed out after 600s."; }
        except OSError as e { return f"Error: could not run command: {str(e)}"; }
        out = (done.stdout or "") + (done.stderr or "");
        if len(out) > 12000 { out = out[:6000] + "\n... [omitted] ...\n" + out[-6000:]; }
        return f"exit code {done.returncode}\n{out}" if out else f"exit code {done.returncode} (no output)";
    }

    def revert_file(path: str) -> str {
        log_call("revert_file", args=path);
        if self._abs(path) is None { return f"Error: '{path}' is outside the repository."; }
        out = self._exec(["git", "checkout", "--", path]);
        return f"OK: reverted {path} to its original contents." if out.startswith("exit code 0") else out;
    }

    def cleanup() {
        scratch = os.path.join(os.path.realpath(self.base_dir), SCRATCH);
        if os.path.isdir(scratch) {
            import shutil;
            shutil.rmtree(scratch, ignore_errors=True);
        }
    }
}
glob repo: Repo = Repo();

sem Repo.read_file = "Read a file from the repository. Files up to 400 lines come back whole; larger ones come back as the requested window. Content has no line-number prefixes, so it can be copied verbatim as an edit anchor.";
sem Repo.read_file.path = "Path relative to the repository root.";
sem Repo.read_file.start_line = "First line to show (1-based).";
sem Repo.read_file.end_line = "Last line to show, inclusive; 0 means to the end.";
sem Repo.grep = "Search file contents with a Python regex. Returns 'path:line: text' per match. Use this to find where a symbol is defined or used.";
sem Repo.grep.pattern = "Python regular expression.";
sem Repo.grep.path = "File or directory to search, relative to the repository root.";
sem Repo.grep.file_glob = "Glob restricting file names, e.g. '*.py'.";
sem Repo.outline = "List every def/class in a file with its line number. Much cheaper than reading the whole file; use it to decide which window to read.";
sem Repo.outline.path = "Path relative to the repository root.";
sem Repo.replace_in_file = "Replace an exact, unique piece of text in a file. Literal match, not regex. Test files are refused. If 'old' is not found or not unique, the file is unchanged and the error says what to do.";
sem Repo.replace_in_file.path = "Path relative to the repository root.";
sem Repo.replace_in_file.old = "Exact existing text, copied character for character including indentation.";
sem Repo.replace_in_file.new = "Replacement text.";
sem Repo.view_diff = "Show the git diff of everything changed so far in this run.";
sem Repo.run_command = "Run one command in the prepared environment, with the working directory already at the repository root: pytest <path>, python <script or -c ...>, or read-only git (status/diff/log/show). This is not a shell: no cd, pipes, redirects, && or environment-variable prefixes -- pass a single command line and the full output comes back with the exit code.";
sem Repo.run_command.command = "The command line, e.g. 'pytest tests/test_x.py -k name'.";
sem Repo.run_snippet = "Run a short Python script (for example a reproduction of the issue) inside the prepared environment. Returns exit code and output.";
sem Repo.run_snippet.code = "Complete Python source of the script.";
sem Repo.revert_file = "Undo every change this run made to one file, restoring the original. Use it when an edit turned out to be in the wrong place.";
sem Repo.revert_file.path = "Path relative to the repository root.";

# ---------------------------------------------------------------------------
# 阶段节点。每个节点：goal + 一个 by-llm 的 work()，tools= 就是这个阶段的能力边界。
# ---------------------------------------------------------------------------
node Plan {
    has goal: str = "Restate the issue as one falsifiable claim (what call misbehaves, what it should do). Use grep/outline to name the 1-3 source files most likely involved. Do not edit.";
    def work(directive: str) -> str by llm(
        tools=[repo.grep, repo.outline, repo.read_file],
        conversation=history, max_react_iterations=6, on_iteration=guard
    );
}

node Explore {
    has goal: str = "Locate the exact function and lines to change. Write a minimal reproduction with run_snippet and confirm it currently fails. Report file, symbol, line range, and the repro output.";
    def work(directive: str) -> str by llm(
        tools=[repo.grep, repo.outline, repo.read_file, repo.run_snippet, repo.view_diff, repo.revert_file],
        conversation=history, max_react_iterations=20, on_iteration=guard
    );
}

node Edit {
    has goal: str = "Apply the smallest fix in library source with replace_in_file (actually call it -- describing an edit is not an edit). Re-run the reproduction with run_snippet; it must now pass. Check view_diff.";
    def work(directive: str) -> str by llm(
        tools=[repo.read_file, repo.grep, repo.outline, repo.replace_in_file, repo.run_snippet, repo.view_diff],
        conversation=history, max_react_iterations=20, on_iteration=guard
    );
}

node Verify {
    has goal: str = "Run the repository's existing tests nearest the changed code (pytest <path>, or the repo's own runner). Report exactly what the output said: passed/failed counts and any failure lines.";
    def work(directive: str) -> str by llm(
        tools=[repo.run_command, repo.view_diff, repo.read_file],
        conversation=history, max_react_iterations=10, on_iteration=guard
    );
}

node Finish {
    has goal: str = "Done.";
}

# 阶段之间的边。capability 是给路由器看的提示：这条边通向的阶段能做什么。
edge PhaseCapability {
    has capability: str;
    has kind: str = "forward";      # forward | repair | relocate | finish.
}
sem PhaseCapability = "A transition to the next phase of the coding task.";
sem PhaseCapability.capability = "What taking this transition accomplishes; choose the edge whose capability matches what the work needs next.";

sem Plan.work = "Carry out the planning phase with the tools provided, then reply with a short summary the next phase needs.";
sem Plan.work.directive = "The goal of this phase.";
sem Explore.work = "Carry out the locating phase with the tools provided, then reply with a short summary the next phase needs.";
sem Explore.work.directive = "The goal of this phase.";
sem Edit.work = "Carry out the editing phase with the tools provided, then reply with a short summary the next phase needs.";
sem Edit.work.directive = "The goal of this phase.";
sem Verify.work = "Carry out the verification phase with the tools provided, then reply with a short summary of the test results.";
sem Verify.work.directive = "The goal of this phase.";
