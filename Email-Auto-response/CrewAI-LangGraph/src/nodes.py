import os
from typing import Any

class Nodes():      
    def __init__(self, mailbox):
        self.mailbox = mailbox  
        self.my_email = os.environ.get('MY_EMAIL') or mailbox.owner_email
    
    def check_email(self, state):
        print("# Checking for new emails")
        emails = self.mailbox.search()
        checked_emails = state.get('checked_emails_ids') or []
        thread = []
        new_emails = []
        for email in emails:
            if (
                    email['id'] not in checked_emails
                    and email['threadId'] not in thread
                    and not (self.my_email and self.my_email in email['sender'])
            ):
                thread.append(email['threadId'])
                new_emails.append(
                    {
                        "id": email['id'],
                        "threadId": email['threadId'],
                        "snippet": email['snippet'],
                        "sender": email["sender"]
                    }
                )
        checked_emails.extend([email['id'] for email in emails])
        return {
            **state,
            "emails": new_emails,
            "checked_emails_ids": checked_emails
        }

    def new_emails(self, state):
        if len(state['emails']) == 0:
            print("## No new emails")
            return "end"
        else:
            print("## New emails")
            return "continue"
