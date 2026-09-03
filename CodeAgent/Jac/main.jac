import from nodes {
    Plan, Explore, Edit, Verify, Finish, PhaseCapability, llm,
    history, tool_log, repo, written, ran_since_edit, TOOL_BUDGET
}

glob MAX_REPAIRS: int = 3;
glob MAX_RELOCATES: int = 1;

glob SYSTEM: str = (
    "You are a coding agent fixing an issue in the repository at the current directory.\n"
    "Edit only library source, never tests. Read before you edit; copy anchor text exactly.\n"
    "Work in phases; each phase tells you its goal and gives you only the tools it needs.\n\n"
    "## Issue\n{issue}"
);

walker CodeAgent {
    has issue: str;
    has repo_root: str = ".";
    has ledger: list[str] = [];
    has repairs: int = 0;
    has relocates: int = 0;
    has answer: str = "";

    def record(title: str, summary: str) {
        self.ledger.append(f"### {title}\n{summary}");
    }

    can setup with Root entry {
        if not [-->] {                      
            pl = Plan(); ex = Explore(); ed = Edit(); ve = Verify(); fi = Finish();
            here ++> pl;
            # Capability is routing hint
            pl +>: PhaseCapability(capability="locate the exact code to change and reproduce the failure") :+> ex;
            ex +>: PhaseCapability(capability="apply the fix to library source and re-run the reproduction") :+> ed;
            ed +>: PhaseCapability(capability="run the repository's own tests against the edited tree") :+> ve;
            ve +>: PhaseCapability(kind="repair", capability="repair: the fix is in the right place but incomplete or wrong -- edit it again") :+> ed;
            ve +>: PhaseCapability(kind="relocate", capability="relocate: the reproduction still fails exactly as before, so the edit is in the wrong place -- revert it and find the real root cause") :+> ex;
            ve +>: PhaseCapability(kind="finish", capability="finish: the reproduction and the covering tests pass after the last edit") :+> fi;
        }
        repo.base_dir = self.repo_root;
        history.clear();
        tool_log.clear();
        history.append({"role": "system", "content": SYSTEM.format(issue=self.issue)});
        visit [-->][?:Plan];                 # 没有 visit，后面的 ability 一个都不会触发
    }

    can planning with Plan entry {
        self.record("Plan", here.work(here.goal));
        visit [->:PhaseCapability:->][?:Explore];              # 计划之后必然要找代码，不用问模型
    }

    can exploring with Explore entry {
        self.record("Explore", here.work(here.goal));
        visit [->:PhaseCapability:->][?:Edit];
    }

    can editing with Edit entry {
        self.record("Edit", here.work(here.goal));
        visit [->:PhaseCapability:->][?:Verify];
    }

    can verifying with Verify entry {
        self.record("Verify", here.work(here.goal));
        # 先走确定性守卫，再问模型 
        if self.repairs >= MAX_REPAIRS or len(tool_log) >= TOOL_BUDGET {
            visit [->:PhaseCapability:->][?:Finish];
            return;
        }
        self.repairs += 1;
        if not written() {
            self.record("Verify", "GUARD: no edit has landed; nothing observed proves anything. Back to Edit.");
            visit [->:PhaseCapability:->][?:Edit];
            return;
        }
        if not ran_since_edit() {
            self.record("Verify", "GUARD: nothing ran after the last edit. Back to Edit: re-run the reproduction and tests.");
            visit [->:PhaseCapability:->][?:Edit];
            return;
        }
        # relocate 这条边只在第一次修复也没救回来、且还没用过时才摆到模型面前。
        if self.repairs >= 2 and self.relocates < MAX_RELOCATES {
            self.relocates += 1;
            visit [edge ->:PhaseCapability:kind != "forward":->] by llm(
                select=1,
                intent="Pick the edge whose capability matches the evidence. 'finish' only if the reproduction and the covering tests pass after the last edit. 'relocate' if the reproduction still fails the same way it did before any edit. 'repair' if the edit changed the behaviour but not enough.",
                incl_info={"progress": "\n\n".join(self.ledger)}
            ) else {
                visit [->:PhaseCapability:->][?:Edit];
            }
            return;
        }
        visit [edge ->:PhaseCapability:kind != "relocate":->] by llm(
            select=1,
            intent="Pick the edge whose capability matches the state of the work. Take the 'finish' edge only if the repository's own tests covering the changed code ran after the last edit and passed, and the reproduction now passes. Otherwise take the 'repair' edge.",
            incl_info={"progress": "\n\n".join(self.ledger)}
        ) else {
            visit [->:PhaseCapability:->][?:Edit];
        }
    }

    can finishing with Finish entry {
        repo.cleanup();                      # 复现脚本不能混进 patch
        self.answer = "\n\n".join(self.ledger);
        report self.answer;
    }
}

def solve(issue: str, repo_root: str = ".") -> str {
    agent = CodeAgent(issue=issue, repo_root=repo_root);
    root spawn agent;
    return agent.answer;
}

with entry:__main__ {
    import sys;
    if len(sys.argv) < 2 {
        print("usage: jac run main.jac <issue text> [repo_root]");
    } else {
        print(solve(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "."));
    }
}
