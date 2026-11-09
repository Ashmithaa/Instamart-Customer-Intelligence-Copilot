#!/usr/bin/env python
# coding: utf-8

# # Part C — Privacy-Safe Ticket Triage

# ##  Load the ticket data and local mock model

# In[1]:


from pathlib import Path
import pandas as pd
import numpy as np
import re
import json
import random
import time
import sys
from IPython.display import display

# Locate the project root
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()

RAW = ROOT / "data" / "raw"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(parents=True, exist_ok=True)

# Keep empty ticket text as an empty string
tickets = pd.read_csv(
    RAW / "tickets.csv",
    keep_default_na=False
)

print("Project root:", ROOT)
print("Tickets shape:", tickets.shape)
print("\nColumns:")
print(tickets.columns.tolist())

print("\nFirst five tickets:")
display(tickets.head())

print("\nEmpty or whitespace-only ticket texts:",
      tickets["text"].astype(str).str.strip().eq("").sum())

print("\nTicket text data type:", tickets["text"].dtype)

# Check the supplied mock LLM module
sys.path.insert(0, str(ROOT))
import mock_llm

print("\nMock LLM module loaded:", mock_llm.__file__)
print("Has reset function:", hasattr(mock_llm, "reset"))
print("Has chat function:", hasattr(mock_llm, "chat"))


# ##  Apply privacy redaction and input guardrails

# In[2]:


# Input guardrail configuration
MAX_INPUT_TOKENS = 400

def est_tokens(text):
    """Estimate tokens using the supplied project's simple word-based estimate."""
    return len(str(text).split())

# Exact redaction patterns specified in the project brief
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_PATTERN = re.compile(r"\b[6-9]\d{9}\b")
ORDER_ID_PATTERN = re.compile(r"ORD-\d{6}")

# Preserve the original text and prepare the redacted text
tickets_work = tickets.copy()
tickets_work["text"] = tickets_work["text"].astype(str)

tickets_work["estimated_tokens"] = tickets_work["text"].apply(est_tokens)

# Guardrails must run before any model call
tickets_work["status"] = "pending"

empty_mask = tickets_work["text"].str.strip().eq("")
long_mask = (
    ~empty_mask &
    (tickets_work["estimated_tokens"] > MAX_INPUT_TOKENS)
)

tickets_work.loc[empty_mask, "status"] = "skipped_empty"
tickets_work.loc[long_mask, "status"] = "flagged_long"

# Redact all tickets for auditing; only eligible redacted text will reach the model
redaction_counts = {"email": 0, "phone": 0, "order_id": 0}

def redact_text(text):
    text, email_count = EMAIL_PATTERN.subn("[EMAIL]", text)
    text, phone_count = PHONE_PATTERN.subn("[PHONE]", text)
    text, order_count = ORDER_ID_PATTERN.subn("[ORDER_ID]", text)

    redaction_counts["email"] += email_count
    redaction_counts["phone"] += phone_count
    redaction_counts["order_id"] += order_count

    return text

tickets_work["redacted_text"] = tickets_work["text"].apply(redact_text)

# Verify no PII patterns remain in the redacted texts
assert not tickets_work["redacted_text"].str.contains(
    EMAIL_PATTERN, regex=True
).any(), "An email address remains after redaction"

assert not tickets_work["redacted_text"].str.contains(
    PHONE_PATTERN, regex=True
).any(), "A phone number remains after redaction"

assert not tickets_work["redacted_text"].str.contains(
    ORDER_ID_PATTERN, regex=True
).any(), "An order ID remains after redaction"

print("Empty tickets:", int(empty_mask.sum()))
print("Long tickets:", int(long_mask.sum()))
print("Eligible tickets for model processing:",
      int((~empty_mask & ~long_mask).sum()))

print("\nRedaction counts:")
print(redaction_counts)
print("Total substitutions:", sum(redaction_counts.values()))
print("PII verification: PASSED")

print("\nStatus counts:")
print(tickets_work["status"].value_counts())

print("\nLong-ticket token estimates:")
display(
    tickets_work.loc[long_mask, ["ticket_id", "estimated_tokens"]]
)


# ##  Create and validate classification prompts

# In[3]:


ZERO_SHOT_SYSTEM = """
CLASSIFY the customer support ticket into exactly one allowed category.

Allowed categories:
- billing
- delivery
- technical
- fraud

Return JSON only, with exactly these keys:
{"category": "...", "reason": "..."}

The category must be one of the four allowed categories.
The reason must briefly explain the classification.
Do not include Markdown, code fences, or text outside the JSON object.
"""

FEW_SHOT_SYSTEM = """
CLASSIFY the customer support ticket into exactly one allowed category.

Allowed categories:
- billing
- delivery
- technical
- fraud

Return JSON only, with exactly these keys:
{"category": "...", "reason": "..."}

The category must be one of the four allowed categories.
The reason must briefly explain the classification.
Do not include Markdown, code fences, or text outside the JSON object.

EXAMPLES:
Ticket: "I was charged twice for the same order."
Output: {"category": "billing", "reason": "The customer reports a duplicate charge."}

Ticket: "My order has not arrived and the delivery is late."
Output: {"category": "delivery", "reason": "The customer reports a delayed delivery."}

Ticket: "The app crashes whenever I try to open it."
Output: {"category": "technical", "reason": "The customer reports an app malfunction."}

Ticket: "Someone accessed my account and placed an order without permission."
Output: {"category": "fraud", "reason": "The customer reports unauthorized account activity."}
"""

# Validate the prompt requirements
assert "CLASSIFY" in ZERO_SHOT_SYSTEM.upper()
assert "EXAMPLES:" not in ZERO_SHOT_SYSTEM
assert "CLASSIFY" in FEW_SHOT_SYSTEM.upper()
assert "EXAMPLES:" in FEW_SHOT_SYSTEM
assert all(category in ZERO_SHOT_SYSTEM for category in
           ["billing", "delivery", "technical", "fraud"])
assert all(category in FEW_SHOT_SYSTEM for category in
           ["billing", "delivery", "technical", "fraud"])
assert '"category"' in ZERO_SHOT_SYSTEM and '"reason"' in ZERO_SHOT_SYSTEM
assert '"category"' in FEW_SHOT_SYSTEM and '"reason"' in FEW_SHOT_SYSTEM

print("Zero-shot prompt validation: PASSED")
print("Few-shot prompt validation: PASSED")
print("\nZero-shot system prompt:\n")
print(ZERO_SHOT_SYSTEM)
print("\nFew-shot system prompt:\n")
print(FEW_SHOT_SYSTEM)


# ##  Test the local mock model

# In[4]:


test_messages = [
    {"role": "system", "content": ZERO_SHOT_SYSTEM},
    {"role": "user", "content": "My order arrived late."}
]

mock_llm.reset()
test_response = mock_llm.chat(test_messages)

print("Response type:", type(test_response))
print("\nResponse content:")
print(test_response)

print("\nToken estimator signature:")
import inspect
print(inspect.signature(mock_llm.est_tokens))

print("\nTransient error class:")
print(mock_llm.TransientLLMError)


# ##  Add safe retries and output validation

# In[5]:


ALLOWED_CATEGORIES = {"billing", "delivery", "technical", "fraud"}
BASE_DELAY = 0.05
MAX_RETRIES = 3

def call_model_with_retries(system_prompt, redacted_text, ticket_id):
    """Call the mock LLM, retrying transient failures up to three times."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": redacted_text}
    ]

    # The user message must contain only the redacted ticket text.
    assert messages[1]["content"] == redacted_text
    assert not EMAIL_PATTERN.search(redacted_text)
    assert not PHONE_PATTERN.search(redacted_text)
    assert not ORDER_ID_PATTERN.search(redacted_text)

    retry_log = []
    for retry_number in range(MAX_RETRIES + 1):
        try:
            response = mock_llm.chat(messages)
            content = response.choices[0].message.content
            tokens_used = response.usage.total_tokens

            return {
                "content": content,
                "tokens_used": tokens_used,
                "retry_log": retry_log,
                "success": True
            }

        except mock_llm.TransientLLMError:
            if retry_number >= MAX_RETRIES:
                return {
                    "content": "",
                    "tokens_used": 0,
                    "retry_log": retry_log,
                    "success": False
                }

            wait_seconds = (
                BASE_DELAY
                * (2 ** retry_number)
                * random.uniform(0.9, 1.15)
            )

            log_entry = {
                "ticket_id": ticket_id,
                "retry_number": retry_number,
                "wait_seconds": wait_seconds
            }
            retry_log.append(log_entry)

            print(
                f"Retry | ticket={ticket_id} | "
                f"retry={retry_number} | wait={wait_seconds:.4f}s"
            )
            time.sleep(wait_seconds)


def validate_model_output(content):
    """Return a validated category, status, and reason."""
    try:
        parsed = json.loads(content)
        category = parsed.get("category")

        if isinstance(category, str) and category in ALLOWED_CATEGORIES:
            return {
                "status": "classified_json",
                "category": category,
                "reason": str(parsed.get("reason", ""))
            }

        # Valid JSON but category is not allowed: do not guess.
        return {
            "status": "flagged_invalid_output",
            "category": "",
            "reason": "JSON category missing or not allowed"
        }

    except (json.JSONDecodeError, AttributeError, TypeError):
        words = re.findall(r"[a-z]+", str(content).lower())
        found_categories = ALLOWED_CATEGORIES.intersection(words)

        if len(found_categories) == 1:
            category = next(iter(found_categories))
            return {
                "status": "classified_keyword",
                "category": category,
                "reason": "Recovered from a single allowed category keyword"
            }

        return {
            "status": "flagged_invalid_output",
            "category": "",
            "reason": "Could not safely validate the model output"
        }


# Validate the response parser with synthetic examples.
assert validate_model_output(
    '{"category": "billing", "reason": "Duplicate charge"}'
)["status"] == "classified_json"

assert validate_model_output(
    "The correct category is delivery"
)["status"] == "classified_keyword"

assert validate_model_output(
    "Could be billing or fraud"
)["status"] == "flagged_invalid_output"

assert validate_model_output(
    '{"category": "unknown", "reason": "test"}'
)["status"] == "flagged_invalid_output"

# Safe synthetic call: no real customer ticket is processed here.
mock_llm.reset()
synthetic_result = call_model_with_retries(
    ZERO_SHOT_SYSTEM,
    "My order arrived late.",
    "SYNTHETIC_TEST"
)
synthetic_validation = validate_model_output(synthetic_result["content"])

print("\nSynthetic call successful:", synthetic_result["success"])
print("Model output:", synthetic_result["content"])
print("Parsed status:", synthetic_validation["status"])
print("Parsed category:", synthetic_validation["category"])
print("Tokens used:", synthetic_result["tokens_used"])
print("\nParser tests: PASSED")


# ##  Run zero-shot classification

# In[6]:


# Run zero-shot classification on eligible tickets only.
mock_llm.reset()

eligible_tickets = tickets_work[
    tickets_work["status"] == "pending"
].copy()

zero_shot_results = []
all_retry_logs = []

for _, ticket in eligible_tickets.iterrows():
    result = call_model_with_retries(
        ZERO_SHOT_SYSTEM,
        ticket["redacted_text"],
        ticket["ticket_id"]
    )

    all_retry_logs.extend(result["retry_log"])

    if result["success"]:
        validation = validate_model_output(result["content"])
        zero_shot_results.append({
            "ticket_id": ticket["ticket_id"],
            "customer_id": ticket["customer_id"],
            "true_category": ticket["true_category"],
            "status": validation["status"],
            "predicted_category": validation["category"],
            "reason": validation["reason"],
            "tokens_used": result["tokens_used"],
            "retries": len(result["retry_log"])
        })
    else:
        zero_shot_results.append({
            "ticket_id": ticket["ticket_id"],
            "customer_id": ticket["customer_id"],
            "true_category": ticket["true_category"],
            "status": "flagged_invalid_output",
            "predicted_category": "",
            "reason": "Transient model errors persisted after retries",
            "tokens_used": 0,
            "retries": len(result["retry_log"])
        })

zero_shot_df = pd.DataFrame(zero_shot_results)

print("\nZero-shot status counts:")
print(zero_shot_df["status"].value_counts())

print("\nTickets processed:", len(zero_shot_df))
print("Retry events logged:", len(all_retry_logs))
print("Tickets requiring retries:", int((zero_shot_df["retries"] > 0).sum()))
print("Tickets with 3 or more retry attempts:", int((zero_shot_df["retries"] >= 3).sum()))

# Invalid outputs count as incorrect predictions.
zero_shot_accuracy = (
    zero_shot_df["predicted_category"]
    .eq(zero_shot_df["true_category"])
    .sum() / len(zero_shot_df)
)

print(f"\nZero-shot accuracy (all eligible tickets): {zero_shot_accuracy:.1%}")
print("\nExample results:")
print(zero_shot_df.head())


# ##  Evaluate zero-shot results

# In[7]:


print("Overall accuracy (invalid outputs count as incorrect):")
print(
    f"{zero_shot_df['predicted_category'].eq(zero_shot_df['true_category']).sum()}"
    f"/{len(zero_shot_df)} = {zero_shot_accuracy:.1%}"
)

valid_mask = zero_shot_df["status"].isin(
    ["classified_json", "classified_keyword"]
)

valid_df = zero_shot_df[valid_mask]

valid_accuracy = (
    valid_df["predicted_category"].eq(valid_df["true_category"]).mean()
)

print(f"\nAccuracy among valid classifications: {valid_accuracy:.1%}")
print(f"Valid classifications: {len(valid_df)}")

print("\nConfusion table (true category vs predicted category):")
print(pd.crosstab(
    zero_shot_df["true_category"],
    zero_shot_df["predicted_category"].replace("", "[INVALID]"),
    margins=True
))

print("\nIncorrect predictions:")
incorrect = zero_shot_df[
    zero_shot_df["predicted_category"].ne(zero_shot_df["true_category"])
]
print(
    incorrect[
        ["ticket_id", "true_category", "predicted_category", "status"]
    ].to_string(index=False)
)


# ##  Run few-shot classification

# In[8]:


#  Few-shot classification

mock_llm.reset()

few_shot_results = []
few_shot_retry_logs = []

for _, ticket in eligible_tickets.iterrows():
    result = call_model_with_retries(
        FEW_SHOT_SYSTEM,
        ticket["redacted_text"],
        ticket["ticket_id"]
    )

    few_shot_retry_logs.extend(result["retry_log"])

    if result["success"]:
        validation = validate_model_output(result["content"])

        few_shot_results.append({
            "ticket_id": ticket["ticket_id"],
            "customer_id": ticket["customer_id"],
            "true_category": ticket["true_category"],
            "status": validation["status"],
            "predicted_category": validation["category"],
            "reason": validation["reason"],
            "tokens_used": result["tokens_used"],
            "retries": len(result["retry_log"])
        })
    else:
        few_shot_results.append({
            "ticket_id": ticket["ticket_id"],
            "customer_id": ticket["customer_id"],
            "true_category": ticket["true_category"],
            "status": "flagged_invalid_output",
            "predicted_category": "",
            "reason": "Transient model errors persisted after retries",
            "tokens_used": 0,
            "retries": len(result["retry_log"])
        })

few_shot_df = pd.DataFrame(few_shot_results)

few_shot_accuracy = (
    few_shot_df["predicted_category"]
    .eq(few_shot_df["true_category"])
    .sum() / len(few_shot_df)
)

print("Few-shot status counts:")
print(few_shot_df["status"].value_counts())

print("\nTickets processed:", len(few_shot_df))
print("Retry events logged:", len(few_shot_retry_logs))
print(f"Few-shot accuracy: {few_shot_accuracy:.1%}")

print("\nFirst five results:")
print(few_shot_df.head())


# ##  Evaluate few-shot results

# In[9]:


#  Inspect few-shot classification errors

correct_count = (
    few_shot_df["predicted_category"]
    .eq(few_shot_df["true_category"])
    .sum()
)

print("Overall few-shot accuracy:")
print(
    f"{correct_count}/{len(few_shot_df)} "
    f"= {correct_count / len(few_shot_df):.1%}"
)

valid_mask = few_shot_df["status"].isin(
    ["classified_json", "classified_keyword"]
)

valid_few_shot_df = few_shot_df[valid_mask].copy()

valid_accuracy = (
    valid_few_shot_df["predicted_category"]
    .eq(valid_few_shot_df["true_category"])
    .mean()
)

print(f"\nAccuracy among valid classifications: {valid_accuracy:.1%}")
print("Valid classifications:", len(valid_few_shot_df))

print("\nStatus counts:")
print(few_shot_df["status"].value_counts())

print("\nIncorrect predictions:")
incorrect_few_shot = few_shot_df[
    few_shot_df["predicted_category"].ne(
        few_shot_df["true_category"]
    )
]

print(
    incorrect_few_shot[
        [
            "ticket_id",
            "true_category",
            "predicted_category",
            "status"
        ]
    ].to_string(index=False)
)


# ##  Demonstrate the token budget

# In[10]:


#  Token-budget demonstration

TOKEN_BUDGET = 6000

mock_llm.reset()

budget_results = []
budget_retry_logs = []
running_tokens = 0

for _, ticket in eligible_tickets.iterrows():

    # Stop before the model call if the budget is already exhausted.
    if running_tokens >= TOKEN_BUDGET:
        budget_results.append({
            "ticket_id": ticket["ticket_id"],
            "customer_id": ticket["customer_id"],
            "true_category": ticket["true_category"],
            "status": "budget_exhausted",
            "predicted_category": "",
            "reason": "Token budget exhausted before model call",
            "tokens_used": 0,
            "retries": 0
        })
        continue

    result = call_model_with_retries(
        ZERO_SHOT_SYSTEM,
        ticket["redacted_text"],
        ticket["ticket_id"]
    )

    budget_retry_logs.extend(result["retry_log"])

    if result["success"]:
        running_tokens += result["tokens_used"]
        validation = validate_model_output(result["content"])

        budget_results.append({
            "ticket_id": ticket["ticket_id"],
            "customer_id": ticket["customer_id"],
            "true_category": ticket["true_category"],
            "status": validation["status"],
            "predicted_category": validation["category"],
            "reason": validation["reason"],
            "tokens_used": result["tokens_used"],
            "retries": len(result["retry_log"])
        })
    else:
        budget_results.append({
            "ticket_id": ticket["ticket_id"],
            "customer_id": ticket["customer_id"],
            "true_category": ticket["true_category"],
            "status": "flagged_invalid_output",
            "predicted_category": "",
            "reason": "Transient model errors persisted after retries",
            "tokens_used": 0,
            "retries": len(result["retry_log"])
        })

budget_demo_df = pd.DataFrame(budget_results)

budget_path = OUTPUTS / "triage_budget_demo.csv"
budget_demo_df.to_csv(budget_path, index=False)

print("Token budget:", TOKEN_BUDGET)
print("Tokens consumed:", running_tokens)
print("\nStatus counts:")
print(budget_demo_df["status"].value_counts())

print("\nBudget-exhausted tickets:",
      int((budget_demo_df["status"] == "budget_exhausted").sum()))

print("\nSaved to:", budget_path)


# ##  Run the final few-shot triage

# In[11]:


#  Final few-shot triage run

TOKEN_BUDGET = 100000

mock_llm.reset()

final_results = []
final_retry_logs = []
running_tokens = 0

for _, ticket in eligible_tickets.iterrows():

    if running_tokens >= TOKEN_BUDGET:
        final_results.append({
            "ticket_id": ticket["ticket_id"],
            "customer_id": ticket["customer_id"],
            "status": "budget_exhausted",
            "category": "",
            "reason": "Token budget exhausted before model call",
            "tokens_used": 0
        })
        continue

    result = call_model_with_retries(
        FEW_SHOT_SYSTEM,
        ticket["redacted_text"],
        ticket["ticket_id"]
    )

    final_retry_logs.extend(result["retry_log"])

    if result["success"]:
        running_tokens += result["tokens_used"]
        validation = validate_model_output(result["content"])

        final_results.append({
            "ticket_id": ticket["ticket_id"],
            "customer_id": ticket["customer_id"],
            "status": validation["status"],
            "category": validation["category"],
            "reason": validation["reason"],
            "tokens_used": result["tokens_used"]
        })
    else:
        final_results.append({
            "ticket_id": ticket["ticket_id"],
            "customer_id": ticket["customer_id"],
            "status": "flagged_invalid_output",
            "category": "",
            "reason": "Transient model errors persisted after retries",
            "tokens_used": 0
        })

final_triage_df = pd.DataFrame(final_results)

print("Final few-shot status counts:")
print(final_triage_df["status"].value_counts())

print("\nTotal tokens consumed:", running_tokens)
print("Retry events:", len(final_retry_logs))
print("Total tickets:", len(final_triage_df))


# ##  Route tickets using customer segments

# In[12]:


#  Customer-segment routing

# Load customer segments created in Part B
segments_df = pd.read_csv(OUTPUTS / "customer_segments.csv")

print("Segment file columns:", segments_df.columns.tolist())
print("Segment rows:", len(segments_df))

# Find the segment-name column
if "Segment" in segments_df.columns:
    segment_col = "Segment"
elif "segment_name" in segments_df.columns:
    segment_col = "segment_name"
else:
    raise ValueError("Could not find the segment-name column.")

# Keep customer ID and segment name
segment_lookup = segments_df[
    ["customer_id", segment_col]
].drop_duplicates("customer_id").rename(
    columns={segment_col: "segment_name"}
)

# Merge segment information into ticket results
final_triage_df = final_triage_df.merge(
    segment_lookup,
    on="customer_id",
    how="left"
)

# Customers without a segment are not considered premium
is_premium = final_triage_df["segment_name"].eq("Premium Customers")

# Default routing
final_triage_df["priority"] = "NORMAL"
final_triage_df["route"] = "support_queue"

# Human review for unsafe or incomplete classifications
human_review_statuses = {
    "skipped_empty",
    "flagged_long",
    "flagged_invalid_output",
    "budget_exhausted"
}

human_mask = final_triage_df["status"].isin(human_review_statuses)
final_triage_df.loc[human_mask, "route"] = "human_review"

# Fraud classifications route to the security team
fraud_mask = (
    ~human_mask
    & final_triage_df["category"].eq("fraud")
)
final_triage_df.loc[fraud_mask, "route"] = "security_team"
final_triage_df.loc[fraud_mask, "priority"] = "HIGH"

# Premium customers with otherwise valid classifications
premium_mask = (
    ~human_mask
    & ~fraud_mask
    & is_premium
)
final_triage_df.loc[premium_mask, "route"] = "premium_desk"
final_triage_df.loc[premium_mask, "priority"] = "HIGH"

# Required final columns and order
final_triage_df = final_triage_df[
    [
        "ticket_id",
        "customer_id",
        "status",
        "category",
        "segment_name",
        "priority",
        "route",
        "tokens_used"
    ]
]

# Save final output
final_path = OUTPUTS / "triage_results.csv"
final_triage_df.to_csv(final_path, index=False)

print("\nFinal status counts:")
print(final_triage_df["status"].value_counts())

print("\nRouting counts:")
print(final_triage_df["route"].value_counts())

print("\nPriority counts:")
print(final_triage_df["priority"].value_counts())

print("\nFinal columns:", final_triage_df.columns.tolist())
print("Final rows:", len(final_triage_df))
print("Saved to:", final_path)

print("\nPreview:")
print(final_triage_df.head(10))


# ##  Add empty and long tickets to the final output

# In[13]:


#  Add empty and long tickets

# Reload the original ticket data and the current final output
all_tickets = pd.read_csv(RAW / "tickets.csv", keep_default_na=False)
final_triage_df = pd.read_csv(OUTPUTS / "triage_results.csv")

# Select tickets excluded by the input guardrails
excluded_tickets = tickets_work[
    tickets_work["status"].isin(["skipped_empty", "flagged_long"])
].copy()

# Prepare the excluded rows using the required final schema
excluded_rows = pd.DataFrame({
    "ticket_id": excluded_tickets["ticket_id"],
    "customer_id": excluded_tickets["customer_id"],
    "status": excluded_tickets["status"],
    "category": "",
    "segment_name": "",
    "priority": "NORMAL",
    "route": "human_review",
    "tokens_used": 0
})

# Combine processed and excluded tickets
final_triage_df = pd.concat(
    [final_triage_df, excluded_rows],
    ignore_index=True
)

# Validate the full ticket set
assert len(final_triage_df) == len(all_tickets) == 164
assert final_triage_df["ticket_id"].nunique() == 164
assert set(final_triage_df["ticket_id"]) == set(all_tickets["ticket_id"])

# Save the complete output
final_path = OUTPUTS / "triage_results.csv"
final_triage_df.to_csv(final_path, index=False)

print("Total original tickets:", len(all_tickets))
print("Final output rows:", len(final_triage_df))
print("Unique ticket IDs:", final_triage_df["ticket_id"].nunique())

print("\nStatus counts:")
print(final_triage_df["status"].value_counts())

print("\nRouting counts:")
print(final_triage_df["route"].value_counts())

print("\nPriority counts:")
print(final_triage_df["priority"].value_counts())

print("\nValidation: PASSED")
print("Saved to:", final_path)


# ##  Validate the final Part C output

# In[14]:


#  Final validation

required_columns = [
    "ticket_id",
    "customer_id",
    "status",
    "category",
    "segment_name",
    "priority",
    "route",
    "tokens_used"
]

triage = pd.read_csv(OUTPUTS / "triage_results.csv", keep_default_na=False)
source_tickets = pd.read_csv(RAW / "tickets.csv", keep_default_na=False)

assert triage.columns.tolist() == required_columns
assert len(triage) == 164
assert triage["ticket_id"].nunique() == 164
assert set(triage["ticket_id"]) == set(source_tickets["ticket_id"])

# Guardrail checks
assert (triage.loc[triage["status"] == "skipped_empty", "route"]
        == "human_review").all()
assert (triage.loc[triage["status"] == "flagged_long", "route"]
        == "human_review").all()

# Invalid outputs must go to human review
assert (triage.loc[
    triage["status"] == "flagged_invalid_output", "route"
] == "human_review").all()

# Valid fraud classifications must go to security with HIGH priority
fraud = triage[
    (triage["status"].isin(["classified_json", "classified_keyword"]))
    & (triage["category"] == "fraud")
]
assert (fraud["route"] == "security_team").all()
assert (fraud["priority"] == "HIGH").all()

# Every token count must be nonnegative
assert (pd.to_numeric(triage["tokens_used"]) >= 0).all()

# No empty or long ticket should have a category
excluded = triage["status"].isin(["skipped_empty", "flagged_long"])
assert triage.loc[excluded, "category"].eq("").all()

print("All Part C final validation checks: PASSED")
print("\nFinal status counts:")
print(triage["status"].value_counts())
print("\nFinal routing counts:")
print(triage["route"].value_counts())
print("\nFile:", OUTPUTS / "triage_results.csv")


# ## Part C — Privacy-Safe Ticket Triage
# 
# ### Privacy and input guardrails
# - Loaded ticket data while preserving empty strings.
# - Skipped 4 empty tickets and flagged 4 tickets exceeding the 400-token input limit.
# - Applied regex redaction for email addresses, phone numbers, and order IDs before model calls.
# - Redaction totals: 55 emails, 91 phone numbers, and 132 order IDs.
# - Validated that sensitive patterns were absent from the redacted ticket messages sent to the model.
# 
# ### Classification and retry handling
# - Compared zero-shot and few-shot system prompts.
# - Required JSON-only responses with an allowed category and reason.
# - Parsed valid JSON responses and used a controlled keyword fallback when JSON parsing failed.
# - Routed invalid model outputs to human review instead of guessing.
# - Recovered 19 transient model errors using exponential backoff with jitter.
# 
# ### Results
# | Metric | Result |
# |---|---:|
# | Original tickets | 164 |
# | Eligible for model calls | 156 |
# | Valid JSON classifications | 135 |
# | Keyword classifications | 12 |
# | Invalid outputs | 9 |
# | Empty tickets skipped | 4 |
# | Long tickets flagged | 4 |
# | Few-shot accuracy among valid classifications | 100% |
# | Few-shot accuracy across all eligible tickets, counting invalid outputs as unsuccessful | 94.2% |
# | Few-shot token usage | 33,268 |
# | Average token usage per eligible ticket | Approximately 213.3 |
# 
# Zero-shot accuracy was 85.7% among valid classifications and 80.8% across all eligible tickets when invalid outputs were counted as unsuccessful.
# 
# ### Routing
# - Invalid, empty, and long tickets were routed to `human_review`.
# - Valid fraud classifications were routed to `security_team` with HIGH priority.
# - Valid tickets belonging to Premium customers were routed to `premium_desk` with HIGH priority.
# - Other valid tickets were routed to `support_queue` with NORMAL priority.
# 
# ### Token-budget demonstration
# The zero-shot budget demonstration used a configured budget of 6,000 tokens and saved `outputs/triage_budget_demo.csv`. Usage reached 6,047 tokens because the implementation checks the budget before each call: a call may begin while usage is still below the limit and then push usage above it. Subsequent calls are blocked.
# 
# ### Output files
# - `outputs/triage_budget_demo.csv`
# - `outputs/triage_results.csv`

# In[ ]:




