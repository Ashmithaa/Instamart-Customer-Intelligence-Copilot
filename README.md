# Instamart Customer Intelligence & Support Copilot

## Overview

This project builds a customer intelligence and support copilot for an Instamart-style grocery delivery platform using a simulated dataset.

The project is divided into four parts:

- **Part A – Data Foundation:** Data cleaning, validation, SQLite database creation, and SQL analysis.
- **Part B – Customer Segmentation & Lapse Prediction:** RFM analysis, K-Means clustering, and machine learning for customer lapse prediction.
- **Part C – Ticket Triage:** Automated support ticket classification, PII redaction, retry handling, and token-budget management.
- **Part D – Policy RAG & Refund Agent:** Policy document retrieval, refund decision-making, tool execution, checkpointing, and crash recovery.

## Reproduce

Run the following commands from the repository root directory.

### Windows

1. Create a virtual environment:

   ```bash
   python -m venv .venv
   ```

2. Activate the environment:

   ```bash
   .venv\Scripts\activate
   ```

3. Upgrade pip and install dependencies:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Generate the project data:

   ```bash
   python generate_data.py
   ```

5. Run Part A – Data Foundation:

   ```bash
   python Part_A.py
   ```

6. Run Part B – Customer Segmentation & Lapse Prediction:

   ```bash
   python Part_B.py
   ```

7. Run Part C – Ticket Triage:

   ```bash
   python Part_C.py
   ```

8. Run Part D – Policy RAG & Refund Agent:

   ```bash
   python Part_D.py
   ```

The four Jupyter notebooks are also included for reviewing the implementation and validation.

## Part A – Data Foundation

### Implementation

- Audited raw customer, product, and order data.
- Removed exact duplicate order records.
- Cleaned customer ages, city names, gender values, and order fields.
- Created cleaned CSV datasets.
- Created a SQLite database with customers, products, and orders tables.
- Implemented seven SQL analysis queries.

### Results

| Metric | Result |
|---|---:|
| Raw customers | 1,200 |
| Raw products | 40 |
| Raw orders | 13,084 |
| Clean customers | 1,200 |
| Clean products | 40 |
| Clean orders | 12,309 |
| Completed orders | 11,110 |
| Cancelled orders | 1,199 |
| Completed-order revenue | INR 2,998,668.80 |
| Average customer spending | INR 2,552.06 |
| SQL queries | 7 |

### SQL query results

| Query | Result rows |
|---|---:|
| Q1 | 30 |
| Q2 | 30 |
| Q3 | 11 |
| Q4 | 25 |
| Q5 | 550 |
| Q6 | 1,175 |
| Q7 | 183 |

**Validation:** All 18 Part A validation checks passed.

### Outputs

- `data/clean/customers.csv`
- `data/clean/products.csv`
- `data/clean/orders.csv`
- `data/instamart.db`
- `sql/queries.sql`

## Part B – Customer Segmentation & Lapse Prediction

### Implementation

- Created customer Recency, Frequency, and Monetary (RFM) features.
- Applied K-Means clustering to identify four customer segments.
- Used PCA to visualize customer segments.
- Compared Logistic Regression and Random Forest for lapse prediction.
- Used hyperparameter tuning and evaluated the models on a held-out test set.
- Generated a retention list containing Premium test customers ranked by predicted lapse probability.
- Saved and reloaded the selected model to verify prediction consistency.

### Customer segments

| Segment | Customers |
|---|---:|
| Occasional | 436 |
| Premium | 138 |
| Frequent Active | 419 |
| At-Risk | 182 |

The RFM analysis contained 1,175 customers. The two PCA components explained approximately 85.23% of the variance.

### Lapse prediction results

| Model | Lapsed-class test F1-score |
|---|---:|
| Logistic Regression | 0.6864 |
| Random Forest | 0.6636 |

Logistic Regression was selected based on its higher test F1-score. Missing a customer who is likely to lapse can mean losing future revenue, while incorrectly targeting a loyal customer can waste retention resources.

The retention list contains up to 10 Premium customers from the test set, ranked by predicted lapse probability.

**Important limitation:** The Premium segment labels use the full RFM period through September 30, 2025, while lapse-model features stop before September 1, 2025. Therefore, the retention list is illustrative and is not a leakage-free prospective evaluation.

**Validation:** All 12 Part B validation checkpoints passed.

### Outputs

- `outputs/customer_segments.csv`
- `outputs/lapse_model.joblib`
- `outputs/retention_list.csv`

## Part C – Ticket Triage

### Implementation

- Built an automated customer support ticket classification workflow.
- Redacted personally identifiable information (PII) before sending ticket text to the model.
- Added input validation to identify empty and excessively long tickets.
- Compared zero-shot and few-shot classification.
- Implemented retry handling with jittered wait times.
- Added token-budget controls.
- Generated ticket triage results and a budget demonstration.

### Results

| Metric | Result |
|---|---:|
| Total tickets | 164 |
| Empty tickets | 4 |
| Long tickets | 4 |
| Eligible tickets | 156 |
| Zero-shot overall accuracy | 80.8% |
| Zero-shot valid-classification accuracy | 85.7% |
| Few-shot overall accuracy | 94.2% |
| Few-shot valid-classification accuracy | 100% |
| Retry events | 19 |
| Demonstration token budget | 6,000 |
| Demonstration tokens consumed | 6,047 |

The few-shot approach achieved 100% accuracy among valid classifications in the tested run.

The budget demonstration stopped processing when the configured token budget was exhausted. Tickets that could not be classified within the available budget were not treated as successful model classifications.

### Routing results

| Routing destination | Tickets |
|---|---:|
| Support queue | 102 |
| Security team | 34 |
| Human review | 17 |
| Premium desk | 11 |

Priority results:

| Priority | Tickets |
|---|---:|
| NORMAL | 119 |
| HIGH | 45 |

Rule validation and routing checks are performed by the program. Invalid, empty, long, or unresolved cases are handled through validation, fallback, or human-review paths rather than being treated as successful automatic classifications.

**Validation:** All Part C final validation checks passed.

### Outputs

- `outputs/triage_results.csv`
- `outputs/triage_budget_demo.csv`

## Part D – Policy RAG & Refund Agent

### Implementation

- Loaded five policy documents from `data/policies/`.
- Split policy documents into chunks of 40 words with 10-word overlap.
- Used TF-IDF vectorization and cosine similarity for policy retrieval.
- Implemented policy-grounded refund decisions.
- Created tools for order lookup, policy search, refund calculation, and refund issuance.
- Added duplicate-refund protection.
- Implemented token, time, and iteration limits with graceful fallback.
- Added checkpoint saving and crash-resume functionality.

### Policy retrieval

The five policy documents produced 15 chunks:

| Policy document | Chunks |
|---|---:|
| Refund policy | 4 |
| Delivery policy | 3 |
| Escalation policy | 3 |
| Payment policy | 3 |
| Privacy policy | 2 |
| **Total** | **15** |

TF-IDF measures how important a word is within a document relative to how common it is across the document collection. A limitation is that it may not recognize two different phrases as having the same meaning when they use different words.

### Refund rules

- Only completed orders are eligible for refunds.
- Completed orders delivered in more than 45 minutes qualify for a full refund.
- Completed orders delivered in 45 minutes or less receive no refund.
- Refunds above INR 1,000 are escalated to a human and are not issued automatically.
- Duplicate refund requests are prevented.

### Seven scenario results

| Scenario | Expected result | Actual result | Status |
|---|---|---|---|
| S1 | REFUND – INR 128.25 | REFUND – INR 128.25 | PASS |
| S2 | REFUND – INR 135.00 | REFUND – INR 135.00 | PASS |
| S3 | ESCALATE | ESCALATE | PASS |
| S4 | DENIED | DENIED | PASS |
| S5 | DENIED | DENIED | PASS |
| S6 | NOT_FOUND | NOT_FOUND | PASS |
| S7 | FALLBACK | FALLBACK_MAX_ITERATIONS | PASS |

**Final scenario validation: 7/7 passed.**

### Stop-safely demonstrations

The agent implements three stopping safeguards:

| Demonstration | Expected status | Result |
|---|---|---|
| Maximum iterations (S7) | FALLBACK_MAX_ITERATIONS | PASS |
| S1 with token_limit = 200 | FALLBACK_TOKEN_LIMIT | PASS |
| S1 with time_limit_seconds = 0 | FALLBACK_TIMEOUT | PASS |

Each fallback returns the completed steps and a human-handoff summary.

### Crash and resume demonstration

A simulated crash was triggered immediately after the second tool execution, after the checkpoint had been saved.

The agent resumed from the saved checkpoint without repeating completed tool executions.

Tools executed across the crash and resume, in order:

1. `get_order`
2. `search_policy`
3. `calculate_refund`
4. `issue_refund`

Final result:

- Refund status: REFUND
- Refund amount: INR 128.25
- Duplicate tool executions: None
- Crash-resume validation: PASSED

### ReAct versus plan-and-execute

The ReAct agent makes decisions step by step, using model calls to observe tool results and decide what to do next. This can require more model calls and tokens, but it allows the agent to respond to unexpected observations, such as a cancelled order or a NOT_FOUND result. A plan-and-execute agent can reduce model calls by generating a complete plan at the beginning, but it may need to replan when tool results differ from expectations. For this refund workflow, ReAct is useful because refund decisions depend on order status, delivery time, policy evidence, and refund amount. Step-by-step decisions make it easier to stop safely or escalate unexpected cases.

**Validation:** All seven final scenario checkpoints passed, the crash-resume test passed, and the execution log structure was valid.

### Outputs

- `outputs/agent_runs.jsonl`
- `outputs/refund_ledger.jsonl`
- `outputs/policy_retrieval_comparison.csv`
- `outputs/policy_fallback_evaluation.csv`
- `outputs/state/FINAL_S1.json` through `FINAL_S7.json`
- `outputs/state/S1_CRASH_RESUME_TEST.json`
- Additional checkpoint and test state files in `outputs/state/`

## Limitations and Next Steps

- The project uses a simulated Instamart-style dataset and mock language-model responses.
- Model performance may change with different data, random seeds, and train-test splits.
- The retention list is illustrative and not a leakage-free prospective evaluation because the Premium segment is derived from the full RFM period.
- The refund agent should be integrated with secure production systems and human approval workflows before real-world deployment.
- Policy retrieval using TF-IDF may miss semantically similar phrases that use different words.
- Future improvements could include real-time data integration, more extensive model evaluation, monitoring, and production-grade security controls.
