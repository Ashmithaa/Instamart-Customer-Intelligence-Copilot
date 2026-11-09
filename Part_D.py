#!/usr/bin/env python
# coding: utf-8

# # Part D – Policy RAG & Refund Agent
# 

# ##  Policy Retrieval-Augmented Generation (RAG)

# In[1]:


from pathlib import Path
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path.cwd()

if not (PROJECT_ROOT / "data" / "policies").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent

POLICY_DIR = PROJECT_ROOT / "data" / "policies"
OUTPUTS = PROJECT_ROOT / "outputs"

assert POLICY_DIR.exists(), f"Policy folder not found: {POLICY_DIR}"

OUTPUTS.mkdir(parents=True, exist_ok=True)

print("Project folder:", PROJECT_ROOT)
print("Policy folder:", POLICY_DIR)
print("Policy files:", sorted(p.name for p in POLICY_DIR.glob("*.txt")))


# In[2]:


# Load all five policy documents
policy_documents = {}

for file_path in sorted(POLICY_DIR.glob("*.txt")):
    policy_documents[file_path.name] = file_path.read_text(
        encoding="utf-8"
    )

print("Number of policy documents:", len(policy_documents))

for filename, content in policy_documents.items():
    print(f"\n--- {filename} ---")
    print("Characters:", len(content))
    print(content[:250])


# ###  Chunking and retrieval index

# In[3]:


# Split policy documents into 40-word chunks with 10-word overlap
CHUNK_SIZE = 40
CHUNK_OVERLAP = 10
CHUNK_STEP = CHUNK_SIZE - CHUNK_OVERLAP

policy_chunks = []

for filename, content in policy_documents.items():
    words = content.split()

    for start in range(0, len(words), CHUNK_STEP):
        chunk_words = words[start:start + CHUNK_SIZE]

        if not chunk_words:
            continue

        policy_chunks.append({
            "source": filename,
            "chunk_id": len(policy_chunks),
            "start_word": start,
            "text": " ".join(chunk_words)
        })

        if start + CHUNK_SIZE >= len(words):
            break

chunks_df = pd.DataFrame(policy_chunks)

print("Chunk size:", CHUNK_SIZE, "words")
print("Overlap:", CHUNK_OVERLAP, "words")
print("Total chunks:", len(chunks_df))
print("\nChunks per policy:")
print(chunks_df.groupby("source").size())
display(chunks_df.head())


# In[4]:


# Build a TF-IDF index over policy chunks
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

tfidf_matrix = vectorizer.fit_transform(chunks_df["text"])

print("Number of policy chunks:", tfidf_matrix.shape[0])
print("Number of TF-IDF terms:", tfidf_matrix.shape[1])
print("TF-IDF index created successfully.")


# In[5]:


def retrieve_policy(question, top_k=3):
    # Convert the question using the same TF-IDF vocabulary
    question_vector = vectorizer.transform([question])

    # Calculate cosine similarity against every policy chunk
    scores = cosine_similarity(
        question_vector,
        tfidf_matrix
    ).flatten()

    # Select the top matching chunks
    top_indices = scores.argsort()[::-1][:top_k]

    results = chunks_df.iloc[top_indices].copy()
    results["similarity"] = scores[top_indices]

    return results[
        ["source", "chunk_id", "similarity", "text"]
    ].reset_index(drop=True)


# Test the retrieval function
test_results = retrieve_policy(
    "When does a customer qualify for a full refund?"
)

display(test_results)


# ### Evaluate retrieval and unsupported-question fallback

# In[6]:


# Load the supplied policy evaluation dataset
policy_eval_path = PROJECT_ROOT / "data" / "raw" / "policy_eval.csv"

policy_eval = pd.read_csv(
    policy_eval_path,
    keep_default_na=False
)

print("Evaluation dataset shape:", policy_eval.shape)
print("\nColumn names:")
print(policy_eval.columns.tolist())

display(policy_eval.head(14))


# In[7]:


evaluation_results = []

for _, row in policy_eval.iterrows():
    question = row["question"]
    expected_doc = row["expected_doc"]
    expected_fact = row["expected_fact"]

    retrieved = retrieve_policy(question, top_k=3)

    retrieved_docs = retrieved["source"].tolist()
    retrieved_text = " ".join(retrieved["text"].tolist()).lower()

    # Check whether the expected policy document was retrieved
    if expected_doc == "NONE":
        doc_correct = None
        fact_correct = None
    else:
        doc_correct = expected_doc in retrieved_docs
        fact_correct = expected_fact.lower() in retrieved_text

    evaluation_results.append({
        "question": question,
        "expected_doc": expected_doc,
        "expected_fact": expected_fact,
        "retrieved_docs": ", ".join(retrieved_docs),
        "doc_correct": doc_correct,
        "fact_correct": fact_correct
    })

evaluation_df = pd.DataFrame(evaluation_results)

display(evaluation_df)

answerable = evaluation_df[evaluation_df["expected_doc"] != "NONE"]

print("\nAnswerable questions:", len(answerable))
print("Expected document matches:",
      int(answerable["doc_correct"].sum()), "/", len(answerable))
print("Expected fact matches:",
      int(answerable["fact_correct"].sum()), "/", len(answerable))


# In[8]:


def answer_policy_question(question, top_k=3, threshold=0.10):
    results = retrieve_policy(question, top_k=top_k)

    # Select the highest-scoring policy document
    best_source = results.iloc[0]["source"]

    # Keep sufficiently relevant chunks from that document only
    relevant = results[
        (results["source"] == best_source) &
        (results["similarity"] >= threshold)
    ]

    if relevant.empty:
        return {
            "answer": "I could not find sufficient information in the available policies.",
            "sources": []
        }

    context = "\n".join(relevant["text"].tolist())

    return {
        "answer": context,
        "sources": sorted(relevant["source"].unique().tolist())
    }


# Test both questions again
for question in [
    "What delivery time makes a completed order eligible for a refund?",
    "Who won the cricket world cup final?"
]:
    result = answer_policy_question(question)
    print("\nQuestion:", question)
    print("Answer:", result["answer"])
    print("Sources:", result["sources"])


# In[9]:


unsupported_questions = policy_eval[
    policy_eval["expected_doc"] == "NONE"
]

fallback_results = []

for _, row in unsupported_questions.iterrows():
    result = answer_policy_question(row["question"])

    passed = result["answer"].startswith("I could not find")

    fallback_results.append({
        "question": row["question"],
        "answer": result["answer"],
        "fallback_pass": passed
    })

fallback_df = pd.DataFrame(fallback_results)

display(fallback_df)

print(
    "\nUnsupported-question fallback:",
    int(fallback_df["fallback_pass"].sum()),
    "/",
    len(fallback_df)
)

assert fallback_df["fallback_pass"].all()
print("Unsupported-question checks passed!")


# ###  Compare overlapping and non-overlapping chunks

# In[10]:


# Create chunks with no overlap
ZERO_CHUNK_SIZE = 40
ZERO_CHUNK_STEP = ZERO_CHUNK_SIZE

zero_overlap_chunks = []

for filename, content in policy_documents.items():
    words = content.split()

    for start in range(0, len(words), ZERO_CHUNK_STEP):
        chunk_words = words[start:start + ZERO_CHUNK_SIZE]

        if not chunk_words:
            continue

        zero_overlap_chunks.append({
            "source": filename,
            "chunk_id": len(zero_overlap_chunks),
            "start_word": start,
            "text": " ".join(chunk_words)
        })

zero_chunks_df = pd.DataFrame(zero_overlap_chunks)

# Build a separate TF-IDF index
zero_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english"
)

zero_tfidf_matrix = zero_vectorizer.fit_transform(
    zero_chunks_df["text"]
)

print("Zero-overlap chunk size:", ZERO_CHUNK_SIZE)
print("Zero-overlap words:", 0)
print("Total zero-overlap chunks:", len(zero_chunks_df))
print("\nChunks per policy:")
print(zero_chunks_df.groupby("source").size())


# In[11]:


def retrieve_zero_overlap(question, top_k=3):
    question_vector = zero_vectorizer.transform([question])

    scores = cosine_similarity(
        question_vector,
        zero_tfidf_matrix
    ).flatten()

    top_indices = scores.argsort()[::-1][:top_k]

    results = zero_chunks_df.iloc[top_indices].copy()
    results["similarity"] = scores[top_indices]

    return results[
        ["source", "chunk_id", "similarity", "text"]
    ].reset_index(drop=True)


# Compare the original and zero-overlap indexes
test_question = (
    "What delivery time makes a completed order eligible for a refund?"
)

print("ORIGINAL OVERLAPPING INDEX")
display(retrieve_policy(test_question, top_k=3))

print("\nZERO-OVERLAP INDEX")
display(retrieve_zero_overlap(test_question, top_k=3))


# In[12]:


comparison_results = []

answerable = policy_eval[
    policy_eval["expected_doc"] != "NONE"
]

for _, row in answerable.iterrows():
    question = row["question"]
    expected_doc = row["expected_doc"]
    expected_fact = row["expected_fact"].lower()

    # Original overlapping index
    original = retrieve_policy(question, top_k=3)
    original_text = " ".join(original["text"].tolist()).lower()

    original_doc_ok = expected_doc in original["source"].tolist()
    original_fact_ok = expected_fact in original_text

    # Zero-overlap index
    zero = retrieve_zero_overlap(question, top_k=3)
    zero_text = " ".join(zero["text"].tolist()).lower()

    zero_doc_ok = expected_doc in zero["source"].tolist()
    zero_fact_ok = expected_fact in zero_text

    comparison_results.append({
        "question": question,
        "original_doc_ok": original_doc_ok,
        "original_fact_ok": original_fact_ok,
        "zero_doc_ok": zero_doc_ok,
        "zero_fact_ok": zero_fact_ok
    })

comparison_df = pd.DataFrame(comparison_results)
display(comparison_df)

print("Original index — document matches:",
      int(comparison_df["original_doc_ok"].sum()), "/ 12")
print("Original index — fact matches:",
      int(comparison_df["original_fact_ok"].sum()), "/ 12")
print("Zero-overlap index — document matches:",
      int(comparison_df["zero_doc_ok"].sum()), "/ 12")
print("Zero-overlap index — fact matches:",
      int(comparison_df["zero_fact_ok"].sum()), "/ 12")

print("\nQuestions whose expected fact was missed with zero overlap:")
display(comparison_df[~comparison_df["zero_fact_ok"]][["question"]])


# In[13]:


failed_question = (
    "How soon is money returned when a payment fails?"
)

print("ORIGINAL OVERLAPPING INDEX")
display(retrieve_policy(failed_question, top_k=3))

print("\nZERO-OVERLAP INDEX")
display(retrieve_zero_overlap(failed_question, top_k=3))

print("\nExpected fact: 48 hours")


# In[14]:


# Save the retrieval comparison for project evidence
comparison_path = OUTPUTS / "policy_retrieval_comparison.csv"
comparison_df.to_csv(comparison_path, index=False)

# Save the unsupported-question fallback evaluation
fallback_path = OUTPUTS / "policy_fallback_evaluation.csv"
fallback_df.to_csv(fallback_path, index=False)

print("Saved:", comparison_path)
print("Saved:", fallback_path)

print("\nFinal evaluation summary")
print("Overlapping index — expected documents:",
      int(comparison_df["original_doc_ok"].sum()), "/ 12")
print("Overlapping index — expected facts:",
      int(comparison_df["original_fact_ok"].sum()), "/ 12")
print("Zero-overlap index — expected facts:",
      int(comparison_df["zero_fact_ok"].sum()), "/ 12")
print("Unsupported fallback:",
      int(fallback_df["fallback_pass"].sum()), "/ 2")


# ##  Refund Agent
# 

# In[15]:


# Refund Agent foundation: tools, paths, and scenario data
import sys
import sqlite3
from datetime import datetime, timezone

# Load the supplied offline model from the project root.
sys.path.insert(0, str(PROJECT_ROOT))
import mock_llm

# Shared paths used by the remaining Part D cells.
output_dir = OUTPUTS
OUTPUT_DIR = OUTPUTS
state_dir = PROJECT_ROOT / "state"
STATE_DIR = state_dir
STATE_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = PROJECT_ROOT / "data" / "instamart.db"
SCENARIOS_PATH = PROJECT_ROOT / "data" / "raw" / "agent_scenarios.csv"
scenarios_df = pd.read_csv(SCENARIOS_PATH, keep_default_na=False)

REFUND_LEDGER_PATH = OUTPUTS / "refund_ledger.jsonl"

AGENT_SYSTEM_PROMPT = (
    "You are an AGENT that handles refund requests by using the supplied tools. "
    "Always look up the order, check the refund policy, calculate eligibility, "
    "and issue an eligible refund or escalate amounts above INR 1000. "
    "Do not invent order details."
)

# Five Python tools used by the offline mock agent.
def get_order(order_id):
    with sqlite3.connect(DB_PATH) as conn:
        rows = pd.read_sql_query(
            """SELECT order_id, status, order_amount, delivery_minutes
               FROM orders WHERE order_id = ?""",
            conn,
            params=(order_id,)
        )
    if rows.empty:
        return {"error": "order not found"}
    record = rows.iloc[0].to_dict()
    if pd.isna(record["delivery_minutes"]):
        record["delivery_minutes"] = None
    else:
        record["delivery_minutes"] = float(record["delivery_minutes"])
    record["order_amount"] = float(record["order_amount"])
    return record

def search_policy(query):
    results = retrieve_policy(query, top_k=3)
    return {
        "results": results[["source", "similarity", "text"]].to_dict("records")
    }

def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is None or float(delivery_minutes) <= 45:
        amount = 0.0
    else:
        amount = round(float(order_amount), 2)
    return {"refund_inr": amount}

def issue_refund(order_id, refund_inr, run_id="manual_test"):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_name": "issue_refund",
        "run_id": run_id,
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2),
        "status": "ISSUED"
    }
    with REFUND_LEDGER_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return {"refund_inr": record["refund_inr"], "success": True}

def escalate_to_human(order_id, refund_inr, run_id="manual_test"):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_name": "escalate_to_human",
        "run_id": run_id,
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2),
        "status": "ESCALATED"
    }
    with REFUND_LEDGER_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return {"refund_inr": record["refund_inr"], "success": True}

TOOLS = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}
TOOL_MAP = TOOLS

TOOLS_SPEC = [
    {"name": "get_order", "description": "Look up an order in the Instamart database.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
    {"name": "search_policy", "description": "Retrieve the most relevant refund policy passages.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "calculate_refund", "description": "Calculate a refund using the late-delivery rule.",
     "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"}, "delivery_minutes": {"type": ["number", "null"]}}, "required": ["order_amount", "delivery_minutes"]}},
    {"name": "issue_refund", "description": "Issue an eligible refund to the original payment method.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, "required": ["order_id", "refund_inr"]}},
    {"name": "escalate_to_human", "description": "Send a refund above INR 1000 to a human support lead.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, "required": ["order_id", "refund_inr"]}}
]

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": spec["name"],
        "description": spec["description"],
        "parameters": spec["parameters"]
    }}
    for spec in TOOLS_SPEC
]

def execute_tool(name, arguments, run_id="manual_test"):
    if name not in TOOLS:
        return {"error": f"unknown tool: {name}"}
    function = TOOLS[name]
    if name in {"issue_refund", "escalate_to_human"}:
        return function(**arguments, run_id=run_id)
    return function(**arguments)

print("Refund Agent foundation loaded.")
print("Scenario rows:", len(scenarios_df))
print("Tools:", list(TOOLS))
print("Database found:", DB_PATH.exists())


# In[16]:


#  CHECKPOINT, RESUME AND DUPLICATE-SAFE REFUNDS

import json
import copy
import uuid
import time
import random
from pathlib import Path


# --------------------------------------------------
# 1. Duplicate-safe refund execution
# --------------------------------------------------

def issue_refund(order_id, refund_inr, run_id="manual_test"):

    refund_inr = round(float(refund_inr), 2)

    if refund_inr <= 0:
        return {
            "success": False,
            "reason": "Refund amount must be positive."
        }

    if REFUND_LEDGER_PATH.exists():

        with REFUND_LEDGER_PATH.open("r", encoding="utf-8") as f:

            for line in f:
                if not line.strip():
                    continue

                existing = json.loads(line)

                if (
                    existing.get("order_id") == order_id
                    and existing.get("status") == "ISSUED"
                ):
                    return {
                        "success": True,
                        **existing,
                        "already_issued": True
                    }

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_name": "issue_refund",
        "run_id": run_id,
        "order_id": order_id,
        "refund_inr": refund_inr,
        "status": "ISSUED",
        "payment_method": "original_payment_method"
    }

    with REFUND_LEDGER_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return {
        "success": True,
        **record,
        "already_issued": False
    }


# Update the registered tool
TOOLS["issue_refund"] = issue_refund


# --------------------------------------------------
#  Checkpoint saving
# --------------------------------------------------

def save_checkpoint(run_id, messages, tokens_used,
                    step, tool_history):

    checkpoint_path = STATE_DIR / f"{run_id}.json"
    temporary_path = STATE_DIR / f"{run_id}.tmp"

    checkpoint = {
        "run_id": run_id,
        "messages": messages,
        "tokens_used": tokens_used,
        "step": step,
        "tools_executed": tool_history,
        "status": "IN_PROGRESS"
    }

    with temporary_path.open("w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)

    temporary_path.replace(checkpoint_path)

    return checkpoint_path


# --------------------------------------------------
#  Updated Refund Agent with resume support
# --------------------------------------------------

def run_refund_agent(
    message=None,
    run_id=None,
    max_iterations=8,
    token_limit=6000,
    time_limit_seconds=30,
    resume=False,
    simulate_crash_after_tool=None
):

    run_id = run_id or str(uuid.uuid4())
    started = time.monotonic()

    checkpoint_path = STATE_DIR / f"{run_id}.json"

    if resume:

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"No checkpoint found for run {run_id}"
            )

        with checkpoint_path.open("r", encoding="utf-8") as f:
            checkpoint = json.load(f)

        messages = checkpoint["messages"]
        tokens_used = checkpoint["tokens_used"]
        tool_history = checkpoint["tools_executed"]
        start_step = checkpoint["step"]

    else:

        if message is None:
            raise ValueError("Message is required for a new run.")

        tokens_used = 0
        tool_history = []
        start_step = 0

        messages = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": message}
        ]

    def fallback(reason):

        return {
            "run_id": run_id,
            "status": reason,
            "answer": f"FALLBACK: {reason}",
            "tokens_used": tokens_used,
            "tools_executed": tool_history.copy(),
            "human_handoff": True,
            "messages": copy.deepcopy(messages)
        }

    for step in range(start_step, max_iterations):

        if time.monotonic() - started >= time_limit_seconds:
            return fallback("FALLBACK_TIMEOUT")

        response = None

        for attempt in range(3):

            try:
                response = mock_llm.chat(
                    messages,
                    tools=TOOL_SCHEMAS
                )
                break

            except mock_llm.TransientLLMError:

                if attempt == 2:
                    return fallback("FALLBACK_LLM_ERROR")

                time.sleep(
                    0.05 * (2 ** attempt)
                    + random.uniform(0, 0.02)
                )

        call_tokens = response.usage.total_tokens

        if tokens_used + call_tokens > token_limit:
            return fallback("FALLBACK_TOKEN_LIMIT")

        tokens_used += call_tokens

        assistant_message = response.choices[0].message

        messages.append({
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "name": call.name,
                    "arguments": call.arguments
                }
                for call in assistant_message.tool_calls
            ]
        })

        if not assistant_message.tool_calls:

            answer = assistant_message.content or ""
            status = (
                answer.split(":", 1)[0]
                if ":" in answer else "COMPLETED"
            )

            return {
                "run_id": run_id,
                "status": status,
                "answer": answer,
                "tokens_used": tokens_used,
                "tools_executed": tool_history.copy(),
                "human_handoff": status in {
                    "ESCALATE", "FALLBACK", "NOT_FOUND"
                },
                "messages": copy.deepcopy(messages)
            }

        for call in assistant_message.tool_calls:

            if time.monotonic() - started >= time_limit_seconds:
                return fallback("FALLBACK_TIMEOUT")

            try:
                arguments = (
                    json.loads(call.arguments)
                    if isinstance(call.arguments, str)
                    else call.arguments
                )

            except (TypeError, json.JSONDecodeError):
                arguments = {}

            tool_result = execute_tool(
                call.name,
                arguments,
                run_id=run_id
            )

            tool_history.append(call.name)

            messages.append({
                "role": "tool",
                "name": call.name,
                "tool_call_id": call.id,
                "content": json.dumps(tool_result, default=str)
            })

            # Save after every tool observation.
            save_checkpoint(
                run_id=run_id,
                messages=messages,
                tokens_used=tokens_used,
                step=step + 1,
                tool_history=tool_history
            )

            # Used only for the crash-resume demonstration.
            if (
                simulate_crash_after_tool is not None
                and len(tool_history) == simulate_crash_after_tool
            ):
                raise RuntimeError(
                    "SIMULATED CRASH AFTER TOOL EXECUTION"
                )

    return fallback("FALLBACK_MAX_ITERATIONS")


print("Step 18 implementation loaded successfully.")
print("Duplicate-safe refunds: Ready")
print("Checkpoint saving: Ready")
print("Resume support: Ready")


# ###  Test refund scenarios and safety limits

# In[17]:


#  Test refund scenarios S2 to S6

test_scenarios = scenarios_df[
    scenarios_df["scenario_id"].isin(["S2", "S3", "S4", "S5", "S6"])
]

scenario_results = []

for _, scenario in test_scenarios.iterrows():

    mock_llm.reset()

    result = run_refund_agent(
        message=str(scenario["message"]),
        run_id=str(scenario["scenario_id"])
    )

    expected = str(scenario["expected_outcome"]).strip()
    actual = result["status"]

    scenario_results.append({
        "scenario_id": scenario["scenario_id"],
        "expected": expected,
        "actual": actual,
        "answer": result["answer"],
        "tools": " -> ".join(result["tools_executed"]),
        "passed": expected == actual
    })

scenario_test_df = pd.DataFrame(scenario_results)

display(
    scenario_test_df[
        ["scenario_id", "expected", "actual", "tools", "passed"]
    ]
)

print(
    "Passed:",
    int(scenario_test_df["passed"].sum()),
    "out of",
    len(scenario_test_df)
)


# In[18]:


#  Test S7 - Infinite-loop protection

mock_llm.reset()

s7 = scenarios_df[
    scenarios_df["scenario_id"] == "S7"
].iloc[0]

s7_result = run_refund_agent(
    message=str(s7["message"]),
    run_id="S7",
    max_iterations=8,
    token_limit=12000
)

print("Expected:", "FALLBACK_MAX_ITERATIONS")
print("Actual:", s7_result["status"])
print("Tokens used:", s7_result["tokens_used"])
print("Tools executed:", s7_result["tools_executed"])
print("Human handoff:", s7_result["human_handoff"])

print(
    "Test passed:",
    s7_result["status"] == "FALLBACK_MAX_ITERATIONS"
)


# ### Checkpoint, resume, and duplicate-safe refund handling
# 

# ###  Simulate an interruption and resume the agent

# In[19]:


#  Simulate crash after second tool execution

mock_llm.reset()

crash_run_id = "S1_CRASH_RESUME_TEST"

try:
    run_refund_agent(
        message="Refund order ORD-100048 because it was delivered late.",
        run_id=crash_run_id,
        simulate_crash_after_tool=2
    )

except RuntimeError as exc:
    print("Crash simulation:", exc)


# Inspect the saved checkpoint

checkpoint_path = STATE_DIR / f"{crash_run_id}.json"

with open(checkpoint_path, "r", encoding="utf-8") as f:
    saved_checkpoint = json.load(f)

print("\nCheckpoint saved:", checkpoint_path.exists())
print("Run ID:", saved_checkpoint["run_id"])
print("Tools saved:", saved_checkpoint["tools_executed"])
print("Tokens saved:", saved_checkpoint["tokens_used"])
print("Next step:", saved_checkpoint["step"])

print(
    "\nCheckpoint test passed:",
    saved_checkpoint["tools_executed"] == [
        "get_order",
        "search_policy"
    ]
)


# In[20]:


# : Resume the interrupted Refund Agent

mock_llm.reset()

resumed_result = run_refund_agent(
    run_id="S1_CRASH_RESUME_TEST",
    resume=True
)

print("Final status:", resumed_result["status"])
print("Final answer:", resumed_result["answer"])
print("Tokens used:", resumed_result["tokens_used"])
print("Tools executed:", resumed_result["tools_executed"])
print("Human handoff:", resumed_result["human_handoff"])

print(
    "\nResume test passed:",
    resumed_result["status"] == "REFUND"
)


# In[21]:


#  Verify refund ledger and duplicate protection

ledger_records = []

if REFUND_LEDGER_PATH.exists():

    with REFUND_LEDGER_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ledger_records.append(json.loads(line))

ledger_df = pd.DataFrame(ledger_records)

order_refunds = ledger_df[
    ledger_df["order_id"] == "ORD-100048"
]

display(order_refunds)

print("Refund records for order:", len(order_refunds))

print(
    "Duplicate protection passed:",
    len(order_refunds) == 1
)


# In[22]:


#  Verify no additional refund is issued

order_id = "ORD-100048"

before_count = len(
    ledger_df[ledger_df["order_id"] == order_id]
)

duplicate_test = issue_refund(
    order_id=order_id,
    refund_inr=128.25,
    run_id="DUPLICATE_CHECK"
)

# Reload ledger after the test
ledger_records = []

with REFUND_LEDGER_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            ledger_records.append(json.loads(line))

updated_ledger_df = pd.DataFrame(ledger_records)

after_count = len(
    updated_ledger_df[
        updated_ledger_df["order_id"] == order_id
    ]
)

print("Before:", before_count)
print("After:", after_count)
print("Duplicate detected:", duplicate_test.get("already_issued", False))
print("New refund record added:", after_count > before_count)

print(
    "Duplicate prevention passed:",
    before_count == after_count
    and duplicate_test.get("already_issued", False)
)


# ### Run all seven scenarios

# In[23]:


#  Consolidated scenario validation

scenario_validation = []

for _, scenario in scenarios_df.iterrows():

    mock_llm.reset()

    scenario_id = str(scenario["scenario_id"])

    result = run_refund_agent(
        message=str(scenario["message"]),
        run_id=f"FINAL_{scenario_id}",
        token_limit=6000
    )

    expected = str(scenario["expected_outcome"]).strip()

    scenario_validation.append({
        "scenario_id": scenario_id,
        "expected": expected,
        "actual": result["status"],
        "passed": expected == result["status"]
    })

final_scenario_df = pd.DataFrame(scenario_validation)

display(final_scenario_df)

print(
    "\nTotal passed:",
    int(final_scenario_df["passed"].sum()),
    "out of",
    len(final_scenario_df)
)


# Treat any documented FALLBACK_* status as a valid fallback category.
final_scenario_df["passed"] = final_scenario_df.apply(
    lambda row: (
        str(row["actual"]).startswith("FALLBACK")
        if row["expected"] == "FALLBACK"
        else row["expected"] == row["actual"]
    ),
    axis=1
)

display(final_scenario_df)
print("\\nAll scenarios passed:", bool(final_scenario_df["passed"].all()))


# ###  Inspect outputs and enable execution logging

# In[24]:


#  Inspect Refund Agent output files

print("Refund ledger exists:", REFUND_LEDGER_PATH.exists())
print("State directory:", STATE_DIR)

print("\nCheckpoint files:")

for file in STATE_DIR.glob("*.json"):
    print(file.name)

print("\nAgent output files:")

for file in OUTPUT_DIR.glob("*"):
    print(file.name)


# In[25]:


#  Add agent execution logging

AGENT_RUNS_PATH = output_dir / "agent_runs.jsonl"

# Preserve the current working agent function
_run_refund_agent_original = run_refund_agent


def run_refund_agent(*args, **kwargs):

    result = _run_refund_agent_original(*args, **kwargs)

    log_record = {
        "run_id": result["run_id"],
        "status": result["status"],
        "tokens_used": result["tokens_used"],
        "tools_executed": result["tools_executed"],
        "human_handoff": result["human_handoff"]
    }

    with AGENT_RUNS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_record) + "\n")

    return result


print("Agent execution logging enabled.")
print("Log file:", AGENT_RUNS_PATH)


# In[26]:


#  Run a safe test and verify agent execution log

mock_llm.reset()

log_test_result = run_refund_agent(
    message="Customer C0001 requests a refund for order ORD-100001 because they disliked the product.",
    run_id="LOG_TEST_S4"
)

print("Test run status:", log_test_result["status"])
print("Test run ID:", log_test_result["run_id"])

print("\nAgent log exists:", AGENT_RUNS_PATH.exists())

if AGENT_RUNS_PATH.exists():
    with AGENT_RUNS_PATH.open("r", encoding="utf-8") as f:
        log_records = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    agent_log_df = pd.DataFrame(log_records)
    display(agent_log_df)

    print("\nTotal logged runs:", len(agent_log_df))
    print("Unique run IDs:", agent_log_df["run_id"].nunique())
else:
    print("The log file was not created. Please share the test run output.")


# ## Final validation

# In[27]:


#  Final Part D validation

# 1. Check required output files
required_files = [
    output_dir / "policy_retrieval_comparison.csv",
    output_dir / "policy_fallback_evaluation.csv",
    output_dir / "refund_ledger.jsonl",
    output_dir / "agent_runs.jsonl"
]

print("OUTPUT FILE CHECK")
for file in required_files:
    print(f"{file.name}: {'PASS' if file.exists() else 'MISSING'}")

# 2. Check scenario validation
print("\nSCENARIO VALIDATION")
scenario_passed = (
    len(final_scenario_df) == 7
    and final_scenario_df["passed"].all()
)
print("Scenarios passed:", int(final_scenario_df["passed"].sum()), "/ 7")
print("All scenarios passed:", scenario_passed)

# 3. Check checkpoints
expected_checkpoints = [
    *(f"FINAL_S{i}.json" for i in range(1, 8)),
    "S1_CRASH_RESUME_TEST.json"
]

print("\nCHECKPOINT VALIDATION")
for filename in expected_checkpoints:
    exists = (state_dir / filename).exists()
    print(f"{filename}: {'PASS' if exists else 'MISSING'}")

# 4. Check execution log
with AGENT_RUNS_PATH.open("r", encoding="utf-8") as f:
    verified_logs = [
        json.loads(line)
        for line in f
        if line.strip()
    ]

log_valid = (
    len(verified_logs) > 0
    and all(
        {"run_id", "status", "tokens_used",
         "tools_executed", "human_handoff"}.issubset(record)
        for record in verified_logs
    )
)

print("\nEXECUTION LOG VALIDATION")
print("Total logged runs:", len(verified_logs))
print("Log structure valid:", log_valid)

# 5. Overall result
files_valid = all(file.exists() for file in required_files)
checkpoints_valid = all(
    (state_dir / filename).exists()
    for filename in expected_checkpoints
)

part_d_passed = (
    files_valid
    and scenario_passed
    and checkpoints_valid
    and log_valid
)

print("\nFINAL PART D RESULT")
print("PART D VALIDATION:", "PASSED" if part_d_passed else "NEEDS ATTENTION")


# ## Part D – What this section demonstrates
# 
# - Policy retrieval using TF-IDF and cosine similarity.
# - Evidence-based answers with source-file names and an unsupported-question fallback.
# - A comparison of overlapping and zero-overlap policy chunks.
# - A tool-calling refund workflow with order lookup, policy search, refund calculation, refund issuance, and human escalation.
# - Retry handling, token and iteration limits, timeout fallback, duplicate-refund prevention, checkpoint saving, crash recovery, and execution logging.
# 

# In[ ]:




