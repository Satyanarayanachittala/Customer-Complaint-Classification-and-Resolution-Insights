"""
Generate synthetic customer complaint dataset for the
Customer Complaint Classification and Resolution Insights project.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define complaint categories and their associated complaint templates
complaint_templates = {
    "Billing": [
        "I was charged twice for my subscription this month. Please refund the extra amount.",
        "My bill shows charges for services I never subscribed to. This is unacceptable.",
        "I cancelled my plan last month but I'm still being billed. Please stop the charges.",
        "The promotional discount was not applied to my latest invoice. I was promised 20% off.",
        "I see an unauthorized charge on my account for $49.99. I did not make this purchase.",
        "My payment was deducted but the service was not activated. I need an immediate refund.",
        "The billing cycle changed without my consent. I want to go back to monthly billing.",
        "I received a late payment fee even though I paid on time. Check your records.",
        "My account shows a balance due but I already paid in full last week.",
        "The price increased without any prior notification. This is unfair to loyal customers.",
        "I was charged for a premium feature that I never opted into. Remove the charge.",
        "My refund has been pending for over 30 days. When will it be processed?",
        "The auto-renewal charged my card without sending a reminder email first.",
        "I see duplicate transactions on my statement. Both need to be investigated.",
        "My corporate discount is not reflecting on the latest bill. Please correct this.",
        "I was charged full price during the free trial period. This is a billing error.",
        "The invoice amount does not match what was quoted by your sales team.",
        "My account was suspended even though the payment went through successfully.",
        "I need an itemized bill breakdown. The total charges seem inflated.",
        "The currency conversion rate applied to my bill is incorrect.",
    ],
    "Technical Support": [
        "The app keeps crashing every time I try to open it. I've tried reinstalling but no luck.",
        "I cannot log into my account. The password reset link is not working either.",
        "The website is extremely slow and keeps timing out. I can't complete any transactions.",
        "My device is not connecting to your service. I've followed all troubleshooting steps.",
        "The software update broke several features that were working fine before.",
        "I'm getting error code 502 whenever I try to access my dashboard.",
        "The search function on your platform returns irrelevant results or no results at all.",
        "Video calls keep dropping after 5 minutes. My internet connection is fine.",
        "The mobile app notifications are not working. I'm missing important alerts.",
        "Data sync between desktop and mobile versions is not working properly.",
        "The export feature generates corrupted files. I cannot open the downloaded reports.",
        "Two-factor authentication is locked and I cannot access my account at all.",
        "The API integration keeps returning timeout errors for our application.",
        "Your software is not compatible with the latest OS update. Please fix this.",
        "The print function is not working correctly. Pages come out blank or garbled.",
        "I cannot upload files larger than 5MB even though the limit should be 25MB.",
        "The chat feature is not loading. I see a blank screen when I click on messages.",
        "Auto-save is not working and I lost 2 hours of work. Very frustrating.",
        "The calendar integration with my email client stopped working after the update.",
        "Screen sharing feature freezes during presentations. This is affecting my work.",
    ],
    "Service Quality": [
        "Your customer service representative was extremely rude and unhelpful during my call.",
        "I've been waiting on hold for over 45 minutes. This is unacceptable wait time.",
        "The service quality has deteriorated significantly over the past few months.",
        "Your team promised a callback within 24 hours but nobody called me back.",
        "The technician who visited my home was unprofessional and left without fixing the issue.",
        "I've contacted support 5 times about the same issue and it's still not resolved.",
        "The quality of your product has declined since I first purchased it.",
        "Your response time to emails is terrible. I waited a week for a simple answer.",
        "The live chat support agent disconnected without resolving my issue.",
        "I was transferred between 4 different departments without anyone helping me.",
        "The service outage lasted 3 days and there was no communication from your team.",
        "Your staff provided incorrect information which caused me financial loss.",
        "The online help documentation is outdated and doesn't match the current interface.",
        "I expect better service for the premium price I'm paying for your subscription.",
        "Your support team keeps asking me to repeat the same information every time I call.",
        "The promised features in the plan I purchased are not available.",
        "Service reliability has been terrible with frequent outages this quarter.",
        "Your company's communication during the maintenance window was poor.",
        "The quality assurance on your latest product release is clearly lacking.",
        "I filed a formal complaint two weeks ago and have received no acknowledgment.",
    ],
    "Delivery": [
        "My order was supposed to arrive 5 days ago but I still haven't received it.",
        "The package arrived damaged. The contents inside were completely broken.",
        "I received the wrong item. The order number matches but the product is different.",
        "The tracking information hasn't updated in 4 days. Where is my package?",
        "My delivery was marked as delivered but I never received it at my address.",
        "The delivery driver left the package in the rain without any protection.",
        "I ordered express shipping but the package is coming via standard delivery.",
        "Half of the items in my order are missing. I only received 2 out of 5 items.",
        "The delivery date was changed twice without notifying me in advance.",
        "My package was returned to sender without any attempt to deliver it.",
        "The fragile items in my order were not packed properly and arrived broken.",
        "I've been waiting 3 weeks for an international shipment that was quoted 7 days.",
        "The delivery person did not ring the doorbell and left a missed delivery notice.",
        "My order shows shipped but the carrier says they never received the package.",
        "The gift wrapping I paid extra for was not included in the delivery.",
        "Perishable items in my order arrived spoiled due to delayed delivery.",
        "The delivery window given was 8am-8pm which is way too broad to be useful.",
        "My replacement order is taking longer than the original order to ship.",
        "The package was left at the wrong address and I had to retrieve it myself.",
        "The estimated delivery date on the website was inaccurate by over a week.",
    ],
    "Account": [
        "I cannot update my email address on the account settings page.",
        "My account was locked without any explanation. I need immediate access.",
        "Someone accessed my account without authorization. I suspect a security breach.",
        "I want to delete my account but there is no option to do so on the website.",
        "My account profile information was changed without my knowledge.",
        "The account merge feature is not working. I have two accounts that need combining.",
        "I cannot change my subscription plan from the account management page.",
        "My loyalty points disappeared from my account after the system update.",
        "I was downgraded from premium to basic without any notification or consent.",
        "The account verification process keeps failing even with correct documents.",
        "I cannot add a secondary user to my account as the option is greyed out.",
        "My account history and past orders have been wiped clean after the migration.",
        "The privacy settings on my account are not saving when I try to update them.",
        "I requested an account data export but received an incomplete file.",
        "My account is showing activity from a location I have never been to.",
        "The account recovery process requires a phone number I no longer have access to.",
        "I cannot link my social media accounts to my profile anymore.",
        "My account tier benefits are not reflecting even though I qualify for gold status.",
        "The parental controls on my account are not restricting content as configured.",
        "I need to transfer my account ownership but the process is not documented.",
    ],
}

# Priority distribution (weighted)
priorities = ["High", "Medium", "Low"]
priority_weights = {
    "Billing": [0.3, 0.5, 0.2],
    "Technical Support": [0.35, 0.45, 0.2],
    "Service Quality": [0.25, 0.45, 0.3],
    "Delivery": [0.3, 0.4, 0.3],
    "Account": [0.4, 0.4, 0.2],
}

# Status distribution
statuses = ["Open", "In Progress", "Closed"]
status_weights = [0.3, 0.25, 0.45]

# Generate dataset
num_records = 1000
data = []

start_date = datetime(2023, 1, 1)
end_date = datetime(2024, 12, 31)
date_range = (end_date - start_date).days

for i in range(1, num_records + 1):
    # Select category
    category = random.choice(list(complaint_templates.keys()))

    # Select complaint text
    complaint_text = random.choice(complaint_templates[category])

    # Add slight variations to make data more realistic
    variations = [
        "",
        " Please resolve this urgently.",
        " I need help with this issue.",
        " This has been going on for too long.",
        " I am very disappointed.",
        " Please look into this matter.",
        " Expecting a quick resolution.",
        " This needs immediate attention.",
    ]
    complaint_text += random.choice(variations)

    # Select priority
    priority = np.random.choice(priorities, p=priority_weights[category])

    # Generate random date
    random_days = random.randint(0, date_range)
    date_received = start_date + timedelta(days=random_days)

    # Select status
    status = np.random.choice(statuses, p=status_weights)

    data.append(
        {
            "complaint_id": i,
            "complaint_text": complaint_text,
            "category": category,
            "priority": priority,
            "date_received": date_received.strftime("%Y-%m-%d"),
            "status": status,
        }
    )

# Create DataFrame
df = pd.DataFrame(data)

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df["complaint_id"] = range(1, len(df) + 1)

# Save to CSV
output_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(output_dir, "complaints.csv")
df.to_csv(output_path, index=False)

print(f"Dataset generated successfully!")
print(f"Total records: {len(df)}")
print(f"Saved to: {output_path}")
print(f"\nCategory distribution:")
print(df["category"].value_counts())
print(f"\nPriority distribution:")
print(df["priority"].value_counts())
print(f"\nStatus distribution:")
print(df["status"].value_counts())
