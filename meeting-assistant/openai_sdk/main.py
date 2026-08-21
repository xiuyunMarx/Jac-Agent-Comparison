"""Entry point for the openai-sdk meeting assistant.

Line for line the flow of ../byLLM/main.jac: register the token tracking,
read meeting_notes.txt from the working directory, run the pipeline, print
the tasks, dump the collected mock-tool outputs. The eval harness
(../eval/run.py) invokes this file in an isolated workdir and reads the
tool_outputs.json it leaves behind.
"""

from nodes import MeetingAssistant
from tools import dump_outputs, register_token_tracking


def main() -> None:
    register_token_tracking()
    print("Loading Meeting Notes")
    with open("meeting_notes.txt", "r") as f:
        transcript = f.read()

    print("Kickoff the Meeting Assistant")
    result = MeetingAssistant(transcript=transcript).run()
    print("TASKS:", result.tasks)
    dump_outputs()


if __name__ == "__main__":
    main()
