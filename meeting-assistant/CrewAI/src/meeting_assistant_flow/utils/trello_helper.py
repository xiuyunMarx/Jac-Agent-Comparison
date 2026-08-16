"""Mock replacement for the Trello helper.

No network calls are made, so benchmark runs measure framework overhead
only. Cards are collected in mock_outputs for post-run evaluation.
"""

from typing import List

from meeting_assistant_flow.types import MeetingTask
from meeting_assistant_flow.utils.mock_outputs import mock_trello_board


def create_trello_card(task_title, task_description):
    """
    Mock creating a new card in Trello for the given task.

    :param task_title: Title of the task (would be the title of the Trello card)
    :param task_description: Detailed description (would be the card body)
    :return: The recorded card dict
    """
    card = {"name": task_title, "desc": task_description}
    mock_trello_board.append(card)
    print(f"[mock trello] Task '{task_title}' successfully created in Trello.")
    return card


def save_tasks_to_trello(tasks: List[MeetingTask]):
    """
    Save a list of tasks to the mock Trello board.

    :param tasks: List of MeetingTask objects with 'name' and 'description'
    """
    for task in tasks:
        if task.name and task.description:
            create_trello_card(task.name, task.description)
        else:
            print("Task is missing a title or description. Skipping...")


if __name__ == "__main__":
    tasks = [
        MeetingTask(name="Add Token Count Progress Indicator", description="..."),
        MeetingTask(name="Improve Mobile Responsiveness", description="..."),
    ]
    save_tasks_to_trello(tasks)
    print("Board:", mock_trello_board)
