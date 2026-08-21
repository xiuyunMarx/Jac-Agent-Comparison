import os;
import from jaclang.byllm.lib { Model }
import from tools { create_trello_card, send_message_to_channel, save_tasks_to_csv }

# One model knob for all three arms. byLLM and CrewAI both reach the provider
# through litellm, which cannot infer a provider from a name it has never seen,
# so the prefix goes on here; the openai_sdk arm strips it back off. Unset, all
# three stay on the gpt-4o this was built against.
def _model_name() -> str {
    name = str(os.environ.get("MEETING_MODEL", "") or "gpt-4o");
    if "/" in name {
        return name;
    }
    return f"openai/{name}";
}

# ctx_window: litellm carries no context-length metadata for a locally served
# model, and without it byLLM's auto-compaction misreads the window and fails on
# short histories. OpenAI models report their own, so stating it costs nothing.
glob llm = Model(
    model_name=_model_name(),
    ctx_window=int(os.environ.get("BENCH_CTX", "0"))
);

node LoadMeetingNode {

}

obj MeetingTask {
    has name: str = "";
    has description: str = "";
}
sem MeetingTask.name= "Short, actionable title for the task";
sem MeetingTask.description= "Detailed description of the task: clear instructions, steps to reproduce, and acceptance criteria where applicable";
node TaskNode {
    has name: str;
    has description: str;
}

node GeneratingTasks {

    def analyse_meeting_transcript(transcript: str) -> list[MeetingTask] by llm();
}
sem GeneratingTasks.analyse_meeting_transcript = "Analyze the meeting transcript and break the discussion down into a list of important, well-structured, actionable tasks that a team can follow up on. Document each task thoroughly."

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