# Instamart Customer Intelligence & Support Copilot

## Overview

An internal Customer Intelligence & Support Copilot for a quick-commerce grocery business, built as one repository in four connected Parts:

| Part | What it does | Main outputs |
|---|---|---|
| A | Audits the messy raw exports, cleans them with documented rules, stores them in SQLite and answers business questions in SQL and pandas | `data/clean/*.csv`, `data/instamart.db`, `sql/queries.sql` |
| B | Segments customers with RFM + K-means and predicts which customers will lapse | `outputs/customer_segments.csv`, `outputs/lapse_model.joblib`, `outputs/retention_list.csv` |
| C | Triages support tickets with privacy redaction, input guardrails, retries, output validation, a token budget and segment-aware routing | `outputs/triage_results.csv`, `outputs/triage_budget_demo.csv` |
| D | Answers policy questions with a TF-IDF retrieval assistant and decides refund requests with a tool-calling ReAct agent | `outputs/agent_runs.jsonl`, `outputs/refund_ledger.jsonl`, `state/` |

All data is synthetic and fictional, produced by `generate_data.py`. Every language-model call goes through the supplied offline `mock_llm.py`, so no API key, account, network connection or paid service is used. All monetary values are in Indian Rupees (INR).

**Tech stack:** Python 3.11+, pandas, numpy, scikit-learn, joblib, sqlite3 (standard library).

### Repository layout

```
generate_data.py        supplied, unmodified
mock_llm.py             supplied, unmodified
partA_clean.py          Part A: audit, clean, store (CSV + SQLite)
validate.py             Part A: asserts the reference checkpoints
partA_analysis.py       Part A: runs the 7 SQL queries, pandas cross-check, EDA
partB_segments.py       Part B: RFM, K-means segments, PCA
partB_lapse.py          Part B: lapse model, class balance, retention list
partC_triage.py         Part C: guardrails, redaction, retries, validation, budget, routing
partD_rag.py            Part D: chunking, TF-IDF retrieval, grounded answers, evaluation
partD_agent.py          Part D: tools, ReAct refund agent, limits, checkpoints, crash/resume
sql/queries.sql         Part A: the 7 SQL queries (Q1-Q7)
data/raw/               generated raw data and policy evaluation files
data/policies/          five policy text documents
data/clean/             cleaned customers, products, orders
data/instamart.db       SQLite database (customers, products, orders)
outputs/                Part B, C, D outputs
state/                  Part D agent checkpoints
```

## Reproduce

Run from the repository root in a clean clone. Tested with Python 3.11 and 3.13 on Windows (Python 3.11 or newer).

1. `python -m venv venv`
2. `source venv/bin/activate`  (Windows: `venv\Scripts\activate`)
3. `pip install -r requirements.txt`
4. `python generate_data.py`
5. `python partA_clean.py`
6. `python validate.py`
7. `python partA_analysis.py`
8. `python partB_segments.py`
9. `python partB_lapse.py`
10. `python partC_triage.py`
11. `python partD_rag.py`
12. `python partD_agent.py`

Step 4 must end with exactly `generated: customers=1200 products=40 orders_raw=13084 tickets=164`, and step 6 must print `ALL CHECKPOINTS PASSED`.

## Part A

### A1. Setup

- Isolated environment created with `python -m venv venv`; libraries installed with pip and pinned in `requirements.txt` via `pip freeze`.
- `.gitignore` lists `.env`, `venv/` and `__pycache__/`.
- `generate_data.py` and `mock_llm.py` are committed unmodified.

### A2. Raw data audit

**Shape, data types and missing values**

`customers.csv` — shape (1200, 5)

| Column | dtype | Missing | Missing % |
|---|---|---|---|
| customer_id | object | 0 | 0.00 |
| city | object | 0 | 0.00 |
| signup_date | object | 0 | 0.00 |
| age | float64 | 61 | 5.08 |
| gender | object | 30 | 2.50 |

`products.csv` — shape (40, 4)

| Column | dtype | Missing | Missing % |
|---|---|---|---|
| product_id | object | 0 | 0.00 |
| product_name | object | 0 | 0.00 |
| category | object | 0 | 0.00 |
| price_inr | int64 | 0 | 0.00 |

`orders.csv` — shape (13084, 9)

| Column | dtype | Missing | Missing % |
|---|---|---|---|
| order_id | object | 0 | 0.00 |
| customer_id | object | 0 | 0.00 |
| product_id | object | 0 | 0.00 |
| quantity | int64 | 0 | 0.00 |
| order_ts | object | 0 | 0.00 |
| status | object | 0 | 0.00 |
| discount_pct | int64 | 0 | 0.00 |
| order_amount | object | 53 | 0.41 |
| delivery_minutes | float64 | 2541 | 19.42 |

`order_amount` is read as `object` because it contains text values. pandas reads the 53 `N/A` entries as missing; the other unparseable entries are the text `error`, which is why the non-numeric count below (120) is larger than the missing count (53).

`tickets.csv` (read with `keep_default_na=False`, used in Part C) — shape (164, 5), all columns `object`, 0 missing values. Empty ticket texts stay empty strings rather than becoming NaN.

**Audit figures**

| Check | Value |
|---|---|
| Raw orders.csv rows | 13,084 |
| Exact duplicate rows | 119 |
| Rows with status `pending` | 656 |
| Distinct `order_id` values | 12,309 |
| `order_amount` entries not convertible to a number | 120 |
| Negative `delivery_minutes` | 106 |
| Blank `delivery_minutes` | 2,541 |
| customers: blank `age` | 61 |
| customers: `age` outside 10–100 | 6 |
| customers: distinct raw `city` spellings | 24 |
| customers: distinct non-blank raw `gender` spellings | 10 |
| customers: blank `gender` | 30 |

### A3. Cleaning rules: customers

| Rule | Values changed | Result |
|---|---|---|
| (a) `city`: strip leading/trailing spaces, apply title case | 24 raw spellings collapsed into 6 | Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184 |
| (b) `gender`: map male/Male/M/m/MALE → `male`, female/Female/F/f/FEMALE → `female`, blank → `unknown` | 10 spellings → 2 values; 30 blanks → `unknown` | female 610, male 560, unknown 30 |
| (c1) `age` outside 10–100 treated as missing | 6 ages (280, −4, 0, 150, 999, 200) | — |
| (c2) Missing `age` filled with the median age of the customer's city (`groupby("city").transform("median")`) | 67 ages filled (61 blank + 6 invalid) | 0 missing; every age between 18 and 60 |
| (d) `signup_date` converted to datetime | all 1,200 rows | stored as text `YYYY-MM-DD` |

### A4. Cleaning rules: orders (applied in this order)

| Step | Rule | Rows / values changed |
|---|---|---|
| i | Convert `order_ts` to datetime | all 13,084 rows parsed (0 unparseable) |
| ii | Remove exact duplicate rows | 119 rows removed |
| iii | For each `order_id`, keep only the record with the latest `order_ts` (removes the earlier `pending` copy of an order) | 656 rows removed |
| iv | Convert `order_amount` to numeric (unparseable → missing), then fill as `round(price_inr × quantity × (1 − discount_pct / 100), 2)` | 112 amounts filled |
| v-a | Negative `delivery_minutes` set to missing | 105 values |
| v-b | For `completed` orders, fill missing `delivery_minutes` with the median delivery time of the product category among completed orders (`groupby("category").transform("median")`) | 774 values filled |
| v-c | For `cancelled` orders, `delivery_minutes` is left missing, because a cancelled order was never delivered and imputing a time would invent data | 1,199 left missing |

The audit counted 120 non-numeric amounts and 106 negative delivery times, but steps iv and v changed 112 and 105. The difference is the rows that were exact duplicates or earlier `pending` copies of the same order: they carried the same bad values and were already removed in steps ii and iii.

**Result after cleaning**

| Checkpoint | Value |
|---|---|
| Rows (one per `order_id`) | 12,309 |
| Completed / cancelled | 11,110 / 1,199 |
| Pending rows | 0 |
| Missing `order_amount` | 0 |
| Missing `delivery_minutes` (completed / cancelled) | 0 / 1,199 |
| Sum of `order_amount` over completed orders | INR 2,998,668.80 |

### A5. Storage and validation

- Clean tables written to `data/clean/customers.csv`, `data/clean/products.csv`, `data/clean/orders.csv`.
- Same tables written to the SQLite database `data/instamart.db` as `customers`, `products`, `orders` with `DataFrame.to_sql`. `order_ts` is stored as text `YYYY-MM-DD HH:MM:SS`.
- `validate.py` asserts the row counts of all three database tables, unique `order_id`, status counts, no pending rows, no missing amounts, the delivery-time missing counts, the completed revenue, the six city counts, the three gender counts and the age range. It prints **`ALL CHECKPOINTS PASSED`**.

### A6. SQL analysis (`sql/queries.sql`)

Each query is run from Python with `pd.read_sql_query` against `data/instamart.db`. Only completed orders count as revenue or spend.

| Query | Technique | Question | Result |
|---|---|---|---|
| Q1 | WHERE then HAVING | Customers whose completed orders above INR 500 total more than INR 8,000 | **30 customers** |
| Q2 | Three-table join | Revenue by city and product category | **30 rows**, summing to **INR 2,998,668.80**; largest cell **Bengaluru / Electronics, INR 233,909.45** |
| Q3 | Scalar subquery | Products priced above the average catalogue price | **11 products** |
| Q4 | LEFT JOIN | Customers with no completed order | **25 customers** |
| Q5 | Two CTEs | Customers whose total completed spend is above the average customer total | **550 customers**; average **INR 2,552.06** |
| Q6 | RANK / DENSE_RANK / ROW_NUMBER over `PARTITION BY city ORDER BY total_spend DESC` | Spend ranking within each city | top customers below; **8 rows** where RANK ≠ ROW_NUMBER |
| Q7 | Running total `SUM(revenue) OVER (ORDER BY day)` | Daily completed revenue with cumulative revenue | **183 daily rows**; last cumulative value **INR 2,998,668.80** |

**Q6 — top customer per city**

| City | Customer | Total spend (INR) |
|---|---|---|
| Bengaluru | C0082 | 12,856.20 |
| Chennai | C0930 | 10,151.40 |
| Delhi | C0291 | 13,742.60 |
| Hyderabad | C0743 | 12,730.05 |
| Mumbai | C0693 | 14,766.55 |
| Pune | C0426 | 10,377.10 |

**Why ties make RANK differ from ROW_NUMBER:** when customers in the same city have exactly the same total spend, RANK() gives all of them the same rank, while ROW_NUMBER() still hands out distinct consecutive numbers, so every tied customer after the first gets a RANK lower than its ROW_NUMBER — 8 such rows here.

### A7. pandas cross-check and exploratory analysis

**(a) Pivot table (completed revenue, INR) with `margins=True`**

| City | Dairy | Electronics | Grocery | Personal Care | Snacks | All |
|---|---|---|---|---|---|---|
| Bengaluru | 87,075.05 | 233,909.45 | 110,915.65 | 20,193.85 | 71,549.40 | 523,643.40 |
| Chennai | 78,301.80 | 195,508.80 | 93,817.45 | 27,675.40 | 68,325.70 | 463,629.15 |
| Delhi | 117,271.70 | 158,549.10 | 112,658.05 | 23,471.50 | 90,879.45 | 502,829.80 |
| Hyderabad | 83,731.00 | 228,913.10 | 109,475.40 | 24,800.05 | 81,670.35 | 528,589.90 |
| Mumbai | 75,209.85 | 214,738.40 | 78,970.15 | 38,018.00 | 72,122.10 | 479,058.50 |
| Pune | 90,349.55 | 187,629.25 | 108,456.60 | 34,275.15 | 80,207.50 | 500,918.05 |
| **All** | 531,938.95 | 1,219,248.10 | 614,293.30 | 168,433.95 | 464,754.50 | **2,998,668.80** |

The pivot grand total (INR 2,998,668.80) equals the sum of the 30 Q2 rows from SQL.

**Per-city completed `order_amount` (groupby with named aggregation)**

| City | Sum (INR) | Mean (INR) | Count |
|---|---|---|---|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

**(b) Distribution of completed `order_amount`**

| Statistic | Value |
|---|---|
| Mean | INR 269.91 |
| Median | INR 135.00 |
| Skewness | 4.61 |
| Q1 / Q3 | INR 64.80 / INR 297.00 |
| IQR | INR 232.20 |
| 1.5×IQR fences | [INR −283.50, INR 645.30] |
| Outliers below / above | 0 / 897 |

Of the 897 high outliers, 635 are Electronics, 166 Dairy, 52 Grocery and 44 Personal Care.

**Correlation matrix (completed orders)**

| | order_amount | discount_pct | quantity | delivery_minutes |
|---|---|---|---|---|
| order_amount | 1.000 | −0.095 | 0.087 | −0.003 |
| discount_pct | −0.095 | 1.000 | −0.000 | −0.010 |
| quantity | 0.087 | −0.000 | 1.000 | −0.013 |
| delivery_minutes | −0.003 | −0.010 | −0.013 | 1.000 |

**Findings (under 200 words)**

- **What is this data about?** 11,110 completed quick-commerce orders from 1,175 customers in six cities between April and September 2025, each with an amount, discount, quantity and delivery time.
- **Distributions and statistics:** most orders are small. The mean (INR 269.91) is about twice the median (INR 135.00) and the skewness is 4.61, so the distribution is strongly right-skewed: many cheap grocery and dairy baskets and a long tail of expensive ones.
- **Outliers and anomalies:** 897 orders lie above the upper fence of INR 645.30 and none below the lower fence. About 71% of them (635) are Electronics, so they are genuine high-value purchases, not data errors, and should be kept.
- **Relationships:** all correlations are very weak. The largest in absolute value is order_amount with discount_pct (−0.095): higher-value orders carry slightly smaller discounts. Next is order_amount with quantity (+0.087). Delivery time is essentially unrelated to the other variables. A correlation this small does not establish causation.

## Part B

Scripts: `partB_segments.py` (tasks 1–3) and `partB_lapse.py` (tasks 4–8). Only completed orders describe customer behaviour, except the cancellation-rate feature.

### B1. RFM table (snapshot 2025-10-01 00:00:00)

| Metric | Value |
|---|---|
| Customers with ≥ 1 completed order | 1,175 |
| Recency (days) median / min / max | 19 / 0 / 180 |
| Frequency mean / max | 9.46 / 33 |
| Monetary mean / max | INR 2,552.06 / INR 14,766.55 |

### B2. K-means on standardised RFM (`n_init=10, random_state=42`)

| K | Inertia | Silhouette | Inertia drop from K−1 |
|---|---|---|---|
| 2 | 1820.8 | 0.465 | — |
| 3 | 1268.2 | 0.445 | 552.6 |
| 4 | 743.1 | **0.526** | 525.1 |
| 5 | 617.4 | 0.480 | 125.7 |
| 6 | 503.2 | 0.482 | 114.2 |
| 7 | 408.5 | 0.416 | 94.7 |
| 8 | 366.5 | 0.412 | 42.0 |

**Chosen K = 4.** The elbow is at K=4: inertia falls by more than 500 at each step up to K=4, then only by about 120 per step. K=4 also has the highest silhouette score (0.526), and four segments are few enough for the retention team to run a separate campaign for each.

**Why lower inertia cannot pick K:** inertia is the total squared distance from each customer to its cluster centre, and adding clusters can only shorten those distances, so K=8 always has lower inertia than K=2 (with K equal to the number of customers it would be zero). Inertia therefore always rewards more clusters; K must be chosen where extra clusters stop giving a large improvement (the elbow), together with silhouette and business usefulness.

### B3. Segment profiles, names and PCA (K = 4)

| segment_id | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Name | Premium | Recommended action |
|---|---|---|---|---|---|---|---|
| 1 | 138 | 24.12 | 6.77 | 7,129.70 | Premium Big Spenders | **True** | Priority support, early access to new electronics, premium loyalty perks |
| 2 | 419 | 8.77 | 19.40 | 3,410.02 | Loyal Regulars | False | Subscription / basket-builder offers to grow order value |
| 0 | 436 | 28.86 | 3.77 | 1,057.62 | Occasional Shoppers | False | Personalised reminders and category cross-sell coupons |
| 3 | 182 | 109.04 | 2.20 | 685.99 | Drifting Away | False | Win-back campaign with a time-limited discount |

Cluster numbers are only labels with no order: segment 3 is not "worse" than segment 0. Names are given from each segment's profile, and Premium is the segment with the highest mean monetary value (segment 1).

**PCA (2 components on the scaled RFM matrix):** explained_variance_ratio_ = **0.634** and **0.218**, total **0.852**, so a 2-D plot keeps 85% of the variation.

Output: `outputs/customer_segments.csv` (customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium).

### B4. Lapse problem (cutoff 2025-09-01)

- **Cohort:** customers with at least one completed order before 2025-09-01.
- **Label:** `lapsed = 1` if the customer has no completed order from 2025-09-01 to 2025-09-30.
- **Features (11):** recency, frequency, monetary, average discount_pct, cancellation rate, average delivery_minutes, tenure in days (numeric); city, gender, favourite product category (nominal); age (numeric).
- **No leakage:** features use only orders dated before 2025-09-01 plus the clean customer table. No data from 2025-09-01 onwards entered the features, and the segment labels from B3 were **not** used as features.

| Checkpoint | Value |
|---|---|
| Cohort | 1,156 customers |
| Lapsed | 443 (38.32%) |
| Train / test (`test_size=0.2, stratify=y, random_state=42`) | 924 / 232 |
| Baseline "always not lapsed": test accuracy / lapsed F1 | 61.6% / 0.000 |

### B5. Preprocessing without leakage

A `ColumnTransformer` inside each scikit-learn `Pipeline` applies `StandardScaler` to the numeric columns and `OneHotEncoder(drop="first")` (k−1 columns) to city, gender and favourite category. **No encoder, scaler or resampler saw test rows:** the pipeline is fitted on training rows only (inside GridSearchCV, on the training folds only) and merely applied to the test rows.

| Mean of frequency | Value |
|---|---|
| Learned by the fitted scaler (training rows) | 8.1147 |
| Full cohort | 8.0285 |

They differ because the scaler learned only from the 924 training rows and never saw the 232 test rows. The code asserts that no feature column contains a missing value.

### B6. Tuning and test evaluation (`cv=5, scoring="f1"`)

| Model | Grid | Best parameters | Best CV F1 |
|---|---|---|---|
| LogisticRegression(max_iter=1000) | C ∈ {0.01, 0.1, 1, 10, 100} | C = 100 | 0.651 |
| RandomForest(n_estimators=100, random_state=42) | max_depth ∈ {3, 5, 8} × min_samples_leaf ∈ {5, 10, 20} (45 fits) | max_depth = 8, min_samples_leaf = 10 | 0.681 |

Test set (lapsed class, evaluated only after tuning was finished):

| Model | Confusion matrix [[TN, FP], [FN, TP]] | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline (always not lapsed) | [[143, 0], [89, 0]] | 0.000 | 0.000 | 0.000 |
| Logistic regression | [[111, 32], [31, 58]] | 0.644 | 0.652 | 0.648 |
| **Random forest (selected)** | [[113, 30], [31, 58]] | 0.659 | 0.652 | **0.655** |

Logistic regression's best C here is 100 rather than the reference's 1, because of the extra features; its test F1 (0.648) is of the same order as the reference.

**Model choice:** the random forest is selected. It catches as many lapsing customers as logistic regression (58 of 89, recall 0.652) while wrongly flagging fewer loyal ones (30 false positives against 32), so it has the higher precision and F1. Missing a lapsing customer costs the business that customer's future orders, while wrongly targeting a loyal one costs only a discount voucher, so recall matters more than precision. Both models have the same recall, so the forest's extra precision decides it; if recall must rise further, section B7 shows how.

### B7. Class-balance experiment (training set only)

Lapsed training rows were randomly duplicated (with replacement, `random_state=42`) until both classes had **570** rows; the test set was not changed. The best logistic regression (C=100) was refitted on the balanced data.

| Logistic regression | Precision | Recall | F1 |
|---|---|---|---|
| Un-balanced training | 0.644 | 0.652 | 0.648 |
| Over-sampled training | 0.589 | **0.854** | 0.697 |

**Recall moved most** (0.652 → 0.854): the balanced model finds 76 of the 89 lapsing customers instead of 58, while precision fell slightly (0.644 → 0.589) rather than rising. For the retention team this means far fewer missed lapsers at the cost of sending some extra offers to customers who would have stayed — a good trade when an offer is cheaper than a lost customer.

### B8. Interpretation, persistence and use

**Top 5 random-forest features (feature_importances_)**

| Feature | Importance | Business reading |
|---|---|---|
| frequency | 0.280 | How often a customer orders is the strongest sign of whether they keep ordering |
| monetary | 0.271 | Total spend separates engaged customers from light ones |
| recency | 0.115 | A long gap since the last order signals a customer drifting away |
| avg_delivery | 0.061 | Delivery experience has a small effect on staying |
| cancel_rate | 0.052 | Cancellation history adds a little signal |

**Logistic-regression coefficient signs (standardised features)**

| Feature | Coefficient | Business reading |
|---|---|---|
| frequency | −1.698 | More past orders → much less likely to lapse; regulars are habit-driven |
| recency | +0.040 | More days since the last order → slightly more likely to lapse |
| monetary | +0.285 | For the same number of orders, high spenders (large, infrequent electronics buys) lapse more often |

**Persistence:** the selected pipeline (preprocessing + random forest) is saved with `joblib.dump` to `outputs/lapse_model.joblib`; after `joblib.load`, the reloaded model reproduces the original 232 test predictions exactly (`True`).

**Retention list:** the 40 test-set customers in the Premium segment are ranked by predicted lapse probability, and the top 10 are written to `outputs/retention_list.csv` (customer_id, lapse_probability, actually_lapsed). **1 of the 10 truly lapsed.** The Premium flag comes from the whole-window RFM, which includes the September window that defines the label, so it is a post-hoc whole-window label and this retention list is illustrative, not a leakage-free evaluation.

## Part C

Script: `partC_triage.py`. Tickets are read with `keep_default_na=False` and processed in file order. Every model call goes through the offline `mock_llm.chat`.

### C1. Input guardrails (before any model call)

| Rule | Status | Tickets |
|---|---|---|
| Text empty or only whitespace | `skipped_empty` | 4 |
| `est_tokens(text)` > `MAX_INPUT_TOKENS = 400` | `flagged_long` | 4 (each estimates to 734 tokens) |

These 8 tickets are never sent to the model.

### C2. Personal-information redaction

The three patterns are applied in this order, to all 164 tickets:

| Type | Pattern | Placeholder | Substitutions |
|---|---|---|---|
| Email | `[\w.+-]+@[\w-]+\.[\w.]+` | `[EMAIL]` | 55 |
| Phone | `\b[6-9]\d{9}\b` | `[PHONE]` | 91 |
| Order id | `ORD-\d{6}` | `[ORDER_ID]` | 132 |
| **Total** | | | **278** |

Re-applying the three patterns to the redacted texts finds **0** matches (asserted in code), and an assertion before every model call checks that the prompt contains no email, phone number or order id. The user message is exactly the redacted ticket text and nothing else.

### C3. Prompts

- **Zero-shot** (about 43 estimated tokens): contains `CLASSIFY`, lists billing, delivery, technical and fraud, and requires a JSON-only reply `{"category": "...", "reason": "..."}`. It does not contain `EXAMPLES:`.
- **Few-shot** (about 117 estimated tokens): the same instructions plus an `EXAMPLES:` line and four labelled examples, one per category, including a fraud example.

### C4. Resilient calls

Every model call is wrapped in a retry function: at most 3 retries per ticket, waiting `BASE_DELAY × 2**n × random.uniform(0.9, 1.15)` seconds before retry n, with `BASE_DELAY = 0.05`. Each retry prints one log line with the ticket id, retry number and wait, for example `retry ticket=T0002 retry_number=0 wait=0.0530s (503 server busy)`. `mock_llm.reset()` is called immediately before each full run.

**Result:** each full run over the 156 eligible tickets raised exactly **19** `TransientLLMError`s, and **all 19 were recovered** on the first retry (every wait between 0.045 and 0.0575 seconds, asserted in code). No ticket ended with a permanent error.

### C5. Output validation

1. Parse the reply as JSON; accept it if `category` is one of the four allowed values → `classified_json`.
2. If it is not valid JSON, extract words with `re.findall(r"[a-z]+", text.lower())`; accept it if exactly one allowed category word is present → `classified_keyword`.
3. Otherwise, including valid JSON with a category outside the list (the mock's `"shipping"`), → `flagged_invalid_output`, routed to human review. The pipeline never guesses.

| Status (one full run, 156 eligible tickets) | Count |
|---|---|
| `classified_json` | 135 |
| `classified_keyword` | 12 |
| `flagged_invalid_output` | 9 |

### C6. Zero-shot vs few-shot (no token budget)

| Prompt | Classified | Correct | Accuracy | Total tokens | Tokens per classified ticket |
|---|---|---|---|---|---|
| Zero-shot | 147 | 126 | **85.7%** | 11,116 | 75.6 |
| Few-shot | 147 | 147 | **100.0%** | 22,660 | 154.1 |

**Zero-shot confusion table** (rows = true category, columns = predicted)

| True \ Predicted | billing | delivery | fraud | technical |
|---|---|---|---|---|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | **11** | 0 | 23 | 0 |
| technical | **10** | 0 | 0 | 29 |

Every zero-shot error is a fraud ticket (11) or a technical ticket (10) predicted as billing.

**When the few-shot cost is worth paying:** the few-shot prompt roughly doubles the cost, from 75.6 to 154.1 tokens per classified ticket (11,544 extra tokens for the run), and raises accuracy from 85.7% to 100%, fixing all 21 errors. Those errors are not harmless: with zero-shot, 11 fraud tickets would land in the normal billing queue instead of going to the security team at HIGH priority. Because a missed fraud case costs far more than about 79 extra tokens per ticket, the few-shot prompt is worth paying for here; zero-shot would only be acceptable for low-risk, high-volume traffic where a wrong queue is cheap to fix.

### C7. Token budget and routing

**(a) Budget demo** (`outputs/triage_budget_demo.csv`): the zero-shot pipeline with `TOKEN_BUDGET = 6000`. Before each call, if the running total of `usage.total_tokens` is already at or above the budget, no call is made and the ticket gets `budget_exhausted` (human review). The run stopped calling the model at 6,010 tokens, and **72 tickets ended `budget_exhausted`**.

| Status | Count |
|---|---|
| classified_json | 74 |
| classified_keyword | 6 |
| flagged_invalid_output | 4 |
| budget_exhausted | 72 |
| skipped_empty / flagged_long | 4 / 4 |

**(b) Final results** (`outputs/triage_results.csv`, 164 rows): a few-shot run with `TOKEN_BUDGET = 100000` (not reached; 22,660 tokens used), routed with the segments from Part B:

| Rule (applied in order) | Route | Priority | Tickets |
|---|---|---|---|
| Status `skipped_empty`, `flagged_long`, `flagged_invalid_output` or `budget_exhausted` | `human_review` | NORMAL | 17 |
| Category `fraud` | `security_team` | HIGH | 34 |
| Customer `is_premium` is true | `premium_desk` | HIGH | 11 |
| Everything else | `support_queue` | NORMAL | 102 |

A customer with no row in `outputs/customer_segments.csv` is treated as not premium. Columns: ticket_id, customer_id, status, category, segment_name, priority, route, tokens_used; category and segment_name are blank when not available.

**Rule-validated vs human:** the 147 tickets with status `classified_json` or `classified_keyword` were classified by the model and accepted only after passing the validation rules in C5; these were routed automatically. The other 17 tickets (4 empty, 4 too long, 9 invalid model outputs) were never auto-classified and went to human review.

## Part D

Scripts: `partD_rag.py` (tasks 1–4, the policy assistant) and `partD_agent.py` (tasks 5–9, the refund agent, which imports `retrieve` from `partD_rag.py`). All model calls go through the offline `mock_llm.chat`; no network call or API key is used.

### D1. Chunking (40 words, overlap 10)

Each document is split into words; chunks cover words 0–39, 30–69, 60–99, … and stop after the first chunk that reaches the last word. Every chunk records its source file.

| Document | Chunks |
|---|---|
| refund_policy.txt | 4 |
| delivery_policy.txt | 3 |
| escalation_policy.txt | 3 |
| payment_policy.txt | 3 |
| privacy_policy.txt | 2 |
| **Total** | **15** |

### D2. TF-IDF retrieval

`TfidfVectorizer()` is fitted on the 15 chunk texts; a query is transformed with the same vectorizer and chunks are ranked by `cosine_similarity`. `retrieve(query, k=3)` returns a list of `{source, score, text}`.

For the query `refund policy late delivery`, the top three chunks all come from **refund_policy.txt** (scores 0.2719, 0.2514, 0.2454).

**TF and IDF:** term frequency (TF) measures how often a word appears in a chunk, and inverse document frequency (IDF) measures how rare the word is across all chunks, so words that are frequent in one chunk but rare overall get the highest weight. **Limitation:** TF-IDF matches exact words only, so a question phrased with different words of the same meaning (for example "money back" instead of "refund", or "slow" instead of "late") can miss the right chunk.

### D3. Augment and answer

`answer(question)` retrieves three chunks and sends two messages: a system message containing `POLICY ASSISTANT` that tells the model to answer only from the context and to say it could not find the information otherwise, and a user message of the form:

```
CONTEXT:
[refund_policy.txt]
<chunk text>
[refund_policy.txt]
<chunk text>
[refund_policy.txt]
<chunk text>
QUESTION:
Who must approve a refund above INR 1000?
```

It returns the model's text and the sources used. Example answer: *"A refund above INR 1000 must be approved by a human support lead before money is returned."*

### D4. Evaluation (`data/raw/policy_eval.csv`, 14 questions)

| Setting | Chunks | Expected doc in top 3 (12 answerable) | Expected fact in answer (12 answerable) | Fallback "I could not find…" (2 unanswerable) |
|---|---|---|---|---|
| 40 words, overlap 10 | 15 | **12/12** | **12/12** | **2/2** |
| 40 words, overlap 0 | 13 | 12/12 | **10/12** | 2/2 |

**What breaks with overlap 0:** two answerable questions lose their fact even though the right document is still retrieved.
- *"What delivery time makes a completed order eligible for a full refund?"* The sentence "If a completed order is delivered in more than 45 minutes, the customer receives a full refund…" is cut at the chunk boundary, so the chunk holding "receives a full refund of the order amount" no longer contains "more than 45 minutes", and the answer misses the expected fact.
- *"How soon is money returned when a payment fails but is deducted?"* The sentence is split right before "within 48 hours", so the answer stops at "the amount is returned automatically".

With overlap 10, every sentence that crosses a boundary also appears whole in the next chunk, which is why the overlap is needed.

### D5. Tools and tool map

| Tool | Description | Parameters |
|---|---|---|
| `get_order` | Look up one order in `data/instamart.db` by its order id | `order_id` (string) |
| `search_policy` | Return the three most relevant policy chunks (`retrieve`) under `results` | `query` (string) |
| `calculate_refund` | 100% of `order_amount` if `delivery_minutes` > 45, otherwise 0 | `order_amount`, `delivery_minutes` (numbers) |
| `issue_refund` | Issue a refund up to INR 1000; appends a timestamped line to `outputs/refund_ledger.jsonl` | `order_id`, `refund_inr` |
| `escalate_to_human` | Send a refund above INR 1000 to a human lead; appends a timestamped line to the ledger | `order_id`, `refund_inr` |

Each tool has a name, a one-sentence description and a JSON-style parameter spec in `TOOLS_SPEC`; `TOOL_MAP` maps each name to its Python function, and the loop calls `TOOL_MAP[name](**arguments)`.

### D6. ReAct loop: seven scenarios

`run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10)` starts with a system message containing `AGENT` plus the user message, calls the model, executes each requested tool, appends the assistant and tool messages, and loops until the model gives a final answer. Every run is appended to `outputs/agent_runs.jsonl` (scenario_id, status, final_answer, llm_calls, tokens_used and every step's tool, arguments and observation).

| Scenario | Status | Refund (INR) | Model calls | Tokens used | Tools executed |
|---|---|---|---|---|---|
| S1 | REFUND | 128.25 | 5 | 1,092 | get_order → search_policy → calculate_refund → issue_refund |
| S2 | REFUND | 135.00 | 5 | 1,092 | get_order → search_policy → calculate_refund → issue_refund |
| S3 | ESCALATE | 1,999.00 | 5 | 1,079 | get_order → search_policy → calculate_refund → escalate_to_human |
| S4 | DENIED | 0.00 | 4 | 761 | get_order → search_policy → calculate_refund |
| S5 | DENIED | 0.00 | 3 | 454 | get_order → search_policy |
| S6 | NOT_FOUND | 0.00 | 2 | 150 | get_order |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,234 | search_policy × 6 |

All seven match the expected outcome and refund amount. S4 is an on-time delivery, S5 a cancelled order, and S6 an unknown order id. S7 completes all six model calls under the default `token_limit=6000` and then stops.

### D7. Stopping safely

| Limit | Rule | Demonstration | Status |
|---|---|---|---|
| Max iterations | After 6 model calls without a final answer | S7 (default limits) | **FALLBACK_MAX_ITERATIONS** |
| Token limit | After each call, if total tokens exceed the limit, stop without running the requested tool | S1 with `token_limit=200` (stopped after 3 calls, 471 tokens; get_order and search_policy done, calculate_refund not run) | **FALLBACK_TOKEN_LIMIT** |
| Time limit | At the top of each iteration, stop if `time.monotonic()` elapsed ≥ the limit | S1 with `time_limit_seconds=0` (stopped before the first model call) | **FALLBACK_TIMEOUT** |

Every fallback returns the steps completed so far and a human-handoff summary in `final_answer`, for example: *"FALLBACK_TOKEN_LIMIT: handing over to a human agent. Reason: token limit of 200 exceeded. Steps completed: 2 (tools: get_order, search_policy)."*

### D8. Checkpoints and crash-and-resume

After every tool observation, a checkpoint `state/<run_id>.json` is written with the run id, scenario id, full message list, tokens used, step number, model-call count and the tools already executed. `resume_agent(run_id)` loads it and continues the loop.

**Demonstration with S1:** a simulated crash is raised right after the second tool execution (after its checkpoint is saved), then `resume_agent` continues from step 2.

| Phase | Tools executed |
|---|---|
| Before the crash | 1. get_order, 2. search_policy |
| After resume | 3. calculate_refund, 4. issue_refund |

Exactly four tool executions, none repeated. Final status **REFUND**, refund **INR 128.25**. The ledger (`outputs/refund_ledger.jsonl`) holds the issued refunds for S1, S2 and the resumed S1 run, plus the S3 escalation.

### D9. ReAct versus plan-and-execute

ReAct asks the model what to do after every observation, so the number of model calls equals the number of tools used plus one: 5 calls for a normal refund (S1, about 1,090 tokens), 3 for the cancelled order (S5) and 2 for the unknown order (S6). Because every call resends the growing message history, tokens grow with each step. A plan-and-execute design asks the model once for the full list of steps (look up, search policy, calculate, issue or escalate) and runs them without further model calls, so a normal refund would need about 1–2 calls and far fewer tokens.

The difference shows on unexpected observations. When `get_order` returns NOT_FOUND or a cancelled status, ReAct simply sees it and stops on the next call (2 and 3 calls). A fixed plan would carry on into `calculate_refund` and `issue_refund` with bad data unless the executor detects the error and replans, which costs an extra planning call and needs careful error rules.

For this refund workflow I would choose **ReAct**. The steps are few, the extra cost is small (about 1,000 tokens per refund), and money is involved, so checking every observation before the next action is worth more than saving a few calls. The step limits, token limit, timeout and checkpoints keep its cost and risk bounded. Plan-and-execute would suit a high-volume, fixed workflow where errors are rare and cheap.

## Limitations and Next Steps

- All data is synthetic, so the patterns found reflect the generator's design rather than real customer behaviour.
- Missing ages and delivery times are filled with group medians, which reduces natural variation in those columns.
- The Premium flag is a post-hoc whole-window label computed from orders that include the September lapse window, so the retention list is illustrative and not a leakage-free evaluation. A production version should compute segments from pre-cutoff orders only.
- The lapse model is tested on a single 232-customer split; repeated cross-validation or a later time window would give a more reliable estimate.
- Part C's triage uses a keyword-based mock model; accuracy against a real hosted model would differ, and the redaction patterns cover only emails, Indian mobile numbers and order ids (names or addresses would pass through).
- Part D's retrieval is word-based TF-IDF, so questions using synonyms can miss the right chunk; an embedding-based retriever would handle paraphrases better.
- The refund agent runs on a deterministic mock model that follows a fixed script; a real model could request tools in a different order, so production use would need stricter argument validation and human approval for every issued refund.
