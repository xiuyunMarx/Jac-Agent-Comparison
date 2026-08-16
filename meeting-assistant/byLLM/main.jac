"""Entry point for the byLLM meeting assistant."""

import from nodes { MeetingAssistant }
import from tools { dump_outputs, register_token_tracking }

with entry {
    register_token_tracking();
    print("Loading Meeting Notes");
    f = open("meeting_notes.txt", "r");
    transcript = f.read();
    f.close();

    print("Kickoff the Meeting Assistant");
    result = root spawn MeetingAssistant(transcript=transcript);
    print("TASKS:", result.tasks);
    dump_outputs();
}
