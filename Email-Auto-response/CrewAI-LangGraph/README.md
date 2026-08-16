# CrewAI + LangGraph

## Introduction
This is an example of how to use [CrewAI](https://github.com/joaomdmoura/crewai) with LangGraph to automate the process of checking emails and creating drafts. CrewAI orchestrates autonomous AI agents, enabling them to collaborate and execute complex tasks efficiently.

This version runs against the shared **mock mailbox** (`../mock_mailbox/`) instead of real Gmail: the inbox, thread lookup, web search, and draft creation are all served from a dataset JSON, so runs are reproducible and need no Gmail credentials or Tavily key. Only the crew's LLM calls are real. Based on the original Gmail-backed example by [@joaomdmoura](https://x.com/joaomdmoura) (preserved at `../../crewAI-examples/integrations/CrewAI-LangGraph`).

![High level image](./CrewAI-LangGraph.png)

## Running the Code

- **Configure Environment**: Copy `.env.example` to `.env` and set `OPENAI_API_KEY`
- **Install Dependencies**: `python -m venv .venv && .venv/bin/pip install -r requirements.txt` (Python <3.14; a ready venv may already exist at `.venv`)
- **Execute**: `.venv/bin/python main.py [path/to/dataset.json]` (defaults to `../mock_mailbox/datasets/batch_001.json`)

Captured drafts, tool-call records and the run's LLM token usage/cost are written to `mock_output/results_<case_id>.json` for scoring against the dataset's ground-truth labels; the run also prints a one-line token/cost summary at the end.

## Details & Explanation
- **Key Components**:
	- `./src/graph.py`: Class defining the nodes and edges (single pass: check emails → draft responses → end).
	- `./src/nodes.py`: Function for the email-checking node (reads the mock mailbox).
	- `./src/state.py`: State declaration.
	- `./src/crew/agents.py`: Class defining the CrewAI Agents.
	- `./src/crew/tasks.py`: Class defining the CrewAI Tasks.
	- `./src/crew/crew.py`: Class defining the CrewAI Crew.
	- `./src/crew/tools.py`: Mailbox-backed tools (get thread, web search, create draft).
- The agents, tasks, and prompts are unchanged from the original Gmail version; only the tool backends differ.

## License
This project is released under the MIT License.
