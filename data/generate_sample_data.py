"""
Generates a small, synthetic-but-realistic sample dataset (data/emails.csv)
so the project runs end-to-end out of the box.

NOTE: This is a DEMO dataset for learning the pipeline. For a real project,
replace this with a real dataset such as:
  - SMS Spam Collection (Kaggle)
  - Enron-Spam dataset
  - SpamAssassin public corpus
See README.md for instructions on swapping in real data.
"""

import csv
import random

random.seed(42)

spam_templates = [
    "CONGRATULATIONS! You have WON a ${amount} prize! Click here to claim now: {link}",
    "URGENT: Your account will be suspended. Verify your details immediately at {link}",
    "Get RICH quick! Make ${amount} a day working from home, no experience needed! {link}",
    "FREE {item} for you! Limited time offer, click {link} before it expires!!!",
    "You are pre-approved for a $10000 loan with 0% interest. Apply now: {link}",
    "HOT SINGLES in your area want to meet you tonight! Sign up free: {link}",
    "Your PayPal account has been LIMITED. Confirm your identity here: {link}",
    "Buy cheap {item} now, 90% OFF! Limited stock, order today: {link}",
    "You've been selected to receive a FREE iPhone! Claim within 24 hours: {link}",
    "LOSE 20 POUNDS in 2 weeks with this ONE weird trick! Learn more: {link}",
    "Nigerian prince needs your help transferring ${amount}. Reply now for your share!",
    "WINNER!! As a valued customer you have been selected for a ${amount} cash prize!",
    "Increase your size guaranteed or your money back! Order now {link}",
    "Your computer has a VIRUS. Download our antivirus NOW to fix it: {link}",
    "Act now! Your subscription is about to expire, renew with 70% discount: {link}",
    "Make money fast with crypto! Guaranteed returns of {amount}% daily, join now: {link}",
    "Dear customer, click here to reset your bank password immediately: {link}",
    "You have an unclaimed inheritance of ${amount}. Contact us urgently to process it.",
    "Exclusive deal just for you: {item} at unbelievable price, click {link} now!",
    "Work from home and earn ${amount}/week! No skills required, sign up: {link}",
]

ham_templates = [
    "Hey, are we still on for lunch tomorrow at {time}?",
    "Please find attached the report you asked for regarding {topic}.",
    "Can you send me the notes from yesterday's meeting about {topic}?",
    "Reminder: your appointment is scheduled for {time} next week.",
    "Thanks for your help with the {topic} project, really appreciate it.",
    "I'll be a few minutes late to the {topic} call, please start without me.",
    "Happy birthday! Hope you have a wonderful day.",
    "Could you review the attached document before {time}?",
    "Let's catch up over coffee sometime next week, are you free {time}?",
    "The invoice for last month's services is attached, let me know if questions.",
    "Just checking in to see how the {topic} plans are coming along.",
    "Don't forget we have the team standup at {time} tomorrow.",
    "Here's the recipe you asked for, hope you enjoy making {item}.",
    "Attached is the presentation for the {topic} meeting on Thursday.",
    "Can we reschedule our call to {time}? Something came up.",
    "Thank you for your order, your {item} will be shipped by {time}.",
    "Great seeing you at the {topic} conference, let's stay in touch.",
    "Your flight confirmation for next {time} is attached, safe travels.",
    "I've updated the {topic} spreadsheet, please review when you can.",
    "Looking forward to the {topic} workshop, see you there!",
]

amounts = ["500", "1000", "5000", "10,000", "250", "2500", "50"]
links = ["bit.ly/x92kf", "clck.it/free-prize", "secure-verify-now.com", "win-big-today.net"]
items = ["watches", "sunglasses", "supplements", "gadget", "cake", "birthday gift", "laptop"]
times = ["3pm", "Monday", "next Friday", "9am", "this weekend", "Tuesday morning"]
topics = ["budget", "marketing", "onboarding", "quarterly review", "client project", "product launch"]


def fill(template):
    return template.format(
        amount=random.choice(amounts),
        link=random.choice(links),
        item=random.choice(items),
        time=random.choice(times),
        topic=random.choice(topics),
    )


rows = []
for _ in range(8):
    for t in spam_templates:
        rows.append(("spam", fill(t)))
for _ in range(8):
    for t in ham_templates:
        rows.append(("ham", fill(t)))

random.shuffle(rows)

with open("/home/claude/spam-detector/data/emails.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["label", "text"])
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to data/emails.csv")
