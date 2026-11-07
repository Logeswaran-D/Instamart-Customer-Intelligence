"""Part C - support ticket triage with guardrails, redaction, retries, validation,
a token budget and segment-aware routing.
Run after:  python partB_segments.py   (needs outputs/customer_segments.csv)
Writes:     outputs/triage_budget_demo.csv, outputs/triage_results.csv
"""
import json
import os
import random
import re
import time

import pandas as pd

import mock_llm
from mock_llm import TransientLLMError, est_tokens

os.makedirs("outputs", exist_ok=True)
random.seed(42)                       # makes the jitter values reproducible

MAX_INPUT_TOKENS = 400
MAX_RETRIES = 3
BASE_DELAY = 0.05
ALLOWED = ["billing", "delivery", "technical", "fraud"]

tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False, dtype=str)

# ------------------------------------------------------------ 2. redaction
PATTERNS = [("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
            ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
            ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]")]


def redact(text):
    counts = {}
    for name, pat, repl in PATTERNS:
        text, n = pat.subn(repl, text)
        counts[name] = n
    return text, counts


def has_pii(text):
    return any(pat.search(text) for _, pat, _ in PATTERNS)


red = tickets["text"].apply(redact)
tickets["redacted"] = red.str[0]
pii = pd.DataFrame(list(red.str[1]))
print("=" * 70, "\n2. REDACTION\n" + "=" * 70)
print(f"emails {pii.email.sum()} | phones {pii.phone.sum()} | order ids {pii.order_id.sum()} | "
      f"total {pii.values.sum()}   (ref 55 / 91 / 132 / 278)")
left = sum(has_pii(t) for t in tickets["redacted"])
assert left == 0
print("matches left after re-applying the three patterns:", left)

# ------------------------------------------------------- 1. input guardrails
def guardrail(text):
    if not text.strip():
        return "skipped_empty"
    if est_tokens(text) > MAX_INPUT_TOKENS:
        return "flagged_long"
    return None


g = tickets["text"].apply(guardrail)
print("\n" + "=" * 70, "\n1. INPUT GUARDRAILS\n" + "=" * 70)
print("skipped_empty:", int((g == "skipped_empty").sum()), "| flagged_long:", int((g == "flagged_long").sum()),
      "(ref 4 / 4)")
print("token estimates of long tickets:",
      sorted({est_tokens(t) for t in tickets.loc[g == "flagged_long", "text"]}), "(ref 734)")

# ------------------------------------------------------------- 3. prompts
ZERO_SHOT = ("You are a support ticket triage assistant. CLASSIFY the customer ticket into exactly one "
             "category: billing, delivery, technical, fraud. Reply with JSON only, in the form "
             '{"category": "...", "reason": "..."} and nothing else.')
FEW_SHOT = ZERO_SHOT + """
EXAMPLES:
Ticket: There is an unauthorized transaction on my card. -> {"category": "fraud", "reason": "unauthorized charge"}
Ticket: I was charged twice for my order. -> {"category": "billing", "reason": "double charge"}
Ticket: The app crashes when I open the cart. -> {"category": "technical", "reason": "app crash"}
Ticket: My order arrived very late. -> {"category": "delivery", "reason": "late delivery"}"""
assert "CLASSIFY" in ZERO_SHOT and "EXAMPLES:" not in ZERO_SHOT
assert "CLASSIFY" in FEW_SHOT and "\nEXAMPLES:\n" in FEW_SHOT
print("\nprompt token estimates: zero-shot", est_tokens(ZERO_SHOT), "| few-shot", est_tokens(FEW_SHOT))

# ------------------------------------------------------- 4. resilient calls
def call_with_retry(messages, ticket_id, log):
    for n in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except TransientLLMError as e:
            if n == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** n * random.uniform(0.9, 1.15)
            log.append({"ticket_id": ticket_id, "retry": n, "wait": wait, "error": str(e)})
            print(f"  retry ticket={ticket_id} retry_number={n} wait={wait:.4f}s ({e})")
            time.sleep(wait)


# ------------------------------------------------------- 5. validate output
def validate(reply):
    try:
        data = json.loads(reply)
        cat = data.get("category") if isinstance(data, dict) else None
        return ("classified_json", cat) if cat in ALLOWED else ("flagged_invalid_output", None)
    except (json.JSONDecodeError, TypeError):
        found = {w for w in re.findall(r"[a-z]+", (reply or "").lower()) if w in ALLOWED}
        if len(found) == 1:
            return "classified_keyword", found.pop()
        return "flagged_invalid_output", None


# --------------------------------------------------------- pipeline run
def run(system_prompt, token_budget=None, label=""):
    mock_llm.reset()
    log, rows, used_total = [], [], 0
    for t in tickets.itertuples():
        row = {"ticket_id": t.ticket_id, "customer_id": t.customer_id, "status": None,
               "category": "", "tokens_used": 0, "true_category": t.true_category}
        status = guardrail(t.text)
        if status:
            row["status"] = status
        elif token_budget is not None and used_total >= token_budget:
            row["status"] = "budget_exhausted"
        else:
            assert not has_pii(t.redacted), "PII would reach the model"
            msgs = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": t.redacted}]           # exactly the redacted text
            resp = call_with_retry(msgs, t.ticket_id, log)
            used = resp.usage.total_tokens
            used_total += used
            row["tokens_used"] = used
            row["status"], cat = validate(resp.choices[0].message.content)
            row["category"] = cat or ""
        rows.append(row)
    df = pd.DataFrame(rows)
    print(f"[{label}] transient errors: {len(log)} (all recovered) | tokens used: {used_total}")
    return df, log, used_total


def summary(df, total):
    cls = df[df.status.isin(["classified_json", "classified_keyword"])]
    correct = int((cls.category == cls.true_category).sum())
    return {"classified": len(cls), "correct": correct,
            "accuracy_%": round(100 * correct / len(cls), 1),
            "total_tokens": total, "tokens_per_classified": round(total / len(cls), 1)}


print("\n" + "=" * 70, "\n4-6. ZERO-SHOT RUN\n" + "=" * 70)
z, zlog, ztot = run(ZERO_SHOT, label="zero-shot")
print("status counts:", z.status.value_counts().to_dict(), "(ref 135 json / 12 keyword / 9 invalid)")
assert len(zlog) == 19 and all(0.045 <= r["wait"] <= 0.0575 for r in zlog if r["retry"] == 0)

print("\n" + "=" * 70, "\n6. FEW-SHOT RUN\n" + "=" * 70)
f, flog, ftot = run(FEW_SHOT, label="few-shot")
print("status counts:", f.status.value_counts().to_dict())

comp = pd.DataFrame([{"prompt": "zero-shot", **summary(z, ztot)},
                     {"prompt": "few-shot", **summary(f, ftot)}])
print("\n", comp.to_string(index=False), "\n(ref accuracy 85.7% zero-shot, 100% few-shot)")
zc = z[z.status.isin(["classified_json", "classified_keyword"])]
print("\nZero-shot confusion table (rows = true, columns = predicted):")
print(pd.crosstab(zc.true_category, zc.category).to_string())

# ------------------------------------------------- 7a. token budget demo
print("\n" + "=" * 70, "\n7a. TOKEN BUDGET DEMO (zero-shot, TOKEN_BUDGET = 6000)\n" + "=" * 70)
b, _, btot = run(ZERO_SHOT, token_budget=6000, label="budget 6000")
n_exh = int((b.status == "budget_exhausted").sum())
print("budget_exhausted tickets:", n_exh)
assert n_exh >= 1
b.drop(columns=["true_category"]).to_csv("outputs/triage_budget_demo.csv", index=False)

# --------------------------------------------- 7b. final routed results
print("\n" + "=" * 70, "\n7b. FINAL RUN (few-shot, TOKEN_BUDGET = 100000) AND ROUTING\n" + "=" * 70)
final, _, _ = run(FEW_SHOT, token_budget=100000, label="final")
seg = pd.read_csv("outputs/customer_segments.csv").set_index("customer_id")
final["segment_name"] = final.customer_id.map(seg["segment_name"]).fillna("")
premium = final.customer_id.map(seg["is_premium"]).fillna(False).astype(bool)
HUMAN = {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"}


def route(i, r):
    if r.status in HUMAN:
        return "human_review", "NORMAL"
    if r.category == "fraud":
        return "security_team", "HIGH"
    if premium[i]:
        return "premium_desk", "HIGH"
    return "support_queue", "NORMAL"


final[["route", "priority"]] = [route(i, r) for i, r in final.iterrows()]
final.loc[final.status.isin(HUMAN), "category"] = ""
cols = ["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]
final[cols].to_csv("outputs/triage_results.csv", index=False)
print("rows:", len(final), "(ref 164)")
print("status counts:", final.status.value_counts().to_dict())
print("routes:", final.route.value_counts().to_dict())
print("Wrote outputs/triage_budget_demo.csv and outputs/triage_results.csv")
