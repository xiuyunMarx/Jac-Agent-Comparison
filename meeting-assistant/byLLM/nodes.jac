import os;
import from jaclang.byllm.lib { Model }
import from tools { create_trello_card, send_message_to_channel, save_tasks_to_csv }

# One knob for every arm: $BENCH_MODEL is the bare model id (default glm-5.2);
# this side is litellm-routed, so it gets the "openai/" provider prefix and the
# OpenAI-compatible endpoint from $OPENAI_BASE_URL (ollama.com by default).
# byLLM 0.31.1 drops config["base_url"] on the litellm path, so the endpoint is
# handed to litellm through OPENAI_API_BASE instead.
glob BENCH_MODEL: str = os.environ.get("BENCH_MODEL", "glm-5.2");
glob _base_set: bool = (os.environ.setdefault(
    "OPENAI_API_BASE", os.environ.get("OPENAI_BASE_URL", "https://ollama.com/v1")
) is not None);

def _model_name() -> str {
    return BENCH_MODEL if "/" in BENCH_MODEL else f"openai/{BENCH_MODEL}";
}

# ctx_window: litellm carries no context-length metadata for a locally served
# model, and without it byLLM's auto-compaction misreads the window and fails on
# short histories. OpenAI models report their own, so stating it costs nothing.
glob llm = Model(
    model_name=_model_name(),
    api_key=os.environ.get("OPENAI_API_KEY", ""),
    ctx_window=int(os.environ.get("BENCH_CTX", "0"))
);

node LoadMeetingNode {

}

obj MeetingTask {
    has name: str;
    has description: str;
}
sem MeetingTask.name = "Required, non-empty, short, actionable title for the task";
sem MeetingTask.description = "Required, non-empty, detailed description of the task: clear instructions, owner, deadline, steps to reproduce, and acceptance criteria where applicable";
node TaskNode {
    has name: str;
    has description: str;
}

def _tasks_are_valid(tasks: list[MeetingTask]) -> bool {
    for task in tasks {
        if not task.name.strip() or not task.description.strip() {
            return False;
        }
    }
    return True;
}

node GeneratingTasks {

    def analyse_meeting_transcript(transcript: str) -> list[MeetingTask] by llm(
        max_output_retries=3
    );

    def repair_meeting_tasks(
        transcript: str, invalid_output: str
    ) -> list[MeetingTask] by llm(max_output_retries=3);
}
sem GeneratingTasks.analyse_meeting_transcript = "Extract every concrete action item from the meeting transcript. Return only a JSON list of MeetingTask objects. Every object must contain exactly two non-empty string fields: name and description; never return an empty or placeholder task. Keep distinct commitments for different deliverables or owners as separate tasks. Combine discussion steps that belong to the same deliverable into one task, including approval followed by execution, rather than over-splitting them. Return an empty list only when the transcript contains no action items. Document owners, deadlines, requirements, and acceptance criteria stated in the transcript without inventing facts.";
sem GeneratingTasks.analyse_meeting_transcript.transcript = "The complete meeting transcript from which to extract action items";
sem GeneratingTasks.repair_meeting_tasks = "The previous extraction contained one or more empty or malformed tasks. Re-extract every concrete action item from the transcript and return a corrected JSON list of MeetingTask objects. Every object must contain exactly two non-empty string fields: name and description; never return an empty or placeholder task. Keep distinct commitments for different deliverables or owners as separate tasks. Combine steps belonging to the same deliverable into one task rather than over-splitting them. Return an empty list only when the transcript contains no action items. Use the invalid output only to understand what must be corrected, not as a source of facts.";
sem GeneratingTasks.repair_meeting_tasks.transcript = "The complete meeting transcript, which is the sole source of truth";
sem GeneratingTasks.repair_meeting_tasks.invalid_output = "The rejected task list containing empty or malformed fields";

node AddTask {
    def work(tasks: list[MeetingTask]) {
        for task in tasks {
            if task.name and task.description {
                create_trello_card(task.name, task.description);
            } else {
                print("Task is missing a name or description. Skipping...");
            }
        }
    }
}

node Save2CSV {
    def work(tasks: list[MeetingTask]) {
        save_tasks_to_csv([(task.name, task.description) for task in tasks]);
    }
}

node SendNotification {
    def work(tasks: list[MeetingTask]) {
        send_message_to_channel(f"{len(tasks)} New tasks have been added to Trello!");
    }
}


walker MeetingAssistant{
    has transcript: str;
    has tasks: list[MeetingTask] = [];
    can setup with Root entry{
        if not [-->] {
            task_gen = GeneratingTasks();
            add_task = AddTask();
            save_csv = Save2CSV();
            send_notification = SendNotification();
            here ++> task_gen;
            task_gen ++> add_task;
            task_gen ++> save_csv;
            task_gen ++> send_notification;
        }
        visit [-->][?:GeneratingTasks];
    }
    can generate_task with GeneratingTasks entry{
        self.tasks = here.analyse_meeting_transcript(self.transcript);
        if not _tasks_are_valid(self.tasks) {
            print("Generated task list contains empty fields. Retrying once with corrective feedback...");
            self.tasks = here.repair_meeting_tasks(
                self.transcript, str(self.tasks)
            );
        }
        if not _tasks_are_valid(self.tasks) {
            raise ValueError(
                "Meeting task extraction still contains empty fields after corrective retry"
            );
        }
        visit [-->];   
    }

    can add_task with AddTask entry{
        here.work(self.tasks);
    }

    can save_csv with Save2CSV entry{
        here.work(self.tasks);
    }

    can send_notification with SendNotification entry{
        here.work(self.tasks);
    }
}
