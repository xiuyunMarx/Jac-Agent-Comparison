"""Mock replacement for the Slack helper.

No network calls are made, so benchmark runs measure framework overhead
only. Messages are collected in mock_outputs for post-run evaluation.
"""

from meeting_assistant_flow.utils.mock_outputs import mock_slack_messages


def send_message_to_channel(text: str):
    mock_slack_messages.append(text)
    print(f"[mock slack] {text}")
    return {"ok": True, "text": text}


if __name__ == "__main__":
    response = send_message_to_channel("Hello, world! This is a test message.")
    print("Message sent successfully!" if response else "Failed to send message.")
