"""Part D (tasks 5-9) - tool-calling ReAct refund agent with limits, fallbacks and checkpoints.
Run after:  python partA_clean.py   (needs data/instamart.db)
Writes:     outputs/agent_runs.jsonl, outputs/refund_ledger.jsonl, state/*.json
"""
import json
import os
import sqlite3
import time
from datetime import datetime, timezone

import pandas as pd

import mock_llm
from partD_rag import retrieve

os.makedirs("outputs", exist_ok=True)
os.makedirs("state", exist_ok=True)
LEDGER, RUNS = "outputs/refund_ledger.jsonl", "outputs/agent_runs.jsonl"
for path in (LEDGER, RUNS):                       # fresh files on every full run
    open(path, "w").close()
for f in os.listdir("state"):
    if f.endswith(".json"):
        os.remove(os.path.join("state", f))

SYSTEM = ("You are a refund AGENT for Instamart support. Use the tools to look up the order, check the "
          "refund policy, calculate the refund and then issue it or escalate it. Finish with one line "
          "starting REFUND, ESCALATE, DENIED or NOT_FOUND.")


# ---------------------------------------------------------------- 5. tools
def get_order(order_id):
    con = sqlite3.connect("data/instamart.db")
    row = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders "
                      "WHERE order_id = ?", (order_id,)).fetchone()
    con.close()
    if row is None:
        return {"error": "order not found"}
    return {"order_id": row[0], "status": row[1], "order_amount": row[2], "delivery_minutes": row[3]}


def search_policy(query):
    return {"results": retrieve(query, k=3)}


def calculate_refund(order_amount, delivery_minutes):
    late = delivery_minutes is not None and delivery_minutes > 45
    return {"refund_inr": round(float(order_amount), 2) if late else 0}


def _ledger(tool, order_id, refund_inr):
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "tool": tool,
                            "order_id": order_id, "refund_inr": refund_inr}) + "\n")
    return {"refund_inr": refund_inr}


def issue_refund(order_id, refund_inr):
    return _ledger("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    return _ledger("escalate_to_human", order_id, refund_inr)


TOOLS_SPEC = [
    {"name": "get_order", "description": "Look up one order in the Instamart database by its order id.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
    {"name": "search_policy", "description": "Return the three policy chunks most relevant to a query.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "calculate_refund", "description": "Compute the refund in INR from the order amount and delivery time.",
     "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"},
                                                     "delivery_minutes": {"type": "number"}},
                    "required": ["order_amount", "delivery_minutes"]}},
    {"name": "issue_refund", "description": "Issue an approved refund of up to INR 1000 and record it in the ledger.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
    {"name": "escalate_to_human", "description": "Send a refund above INR 1000 to a human support lead for approval.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}},
                    "required": ["order_id", "refund_inr"]}},
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}
assert [t["name"] for t in TOOLS_SPEC] == list(TOOL_MAP)


# ------------------------------------------------------- 8. checkpoints
class SimulatedCrash(Exception):
    pass


def save_checkpoint(st):
    with open(os.path.join("state", f"{st['run_id']}.json"), "w", encoding="utf-8") as f:
        json.dump(st, f, indent=1, default=str)


def load_checkpoint(run_id):
    with open(os.path.join("state", f"{run_id}.json"), encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------- 6-7. ReAct loop + limits
def _fallback(st, status, reason):
    done = ", ".join(st["tools_executed"]) or "none"
    st["status"] = status
    st["final_answer"] = (f"{status}: handing over to a human agent. Reason: {reason}. "
                          f"Steps completed: {st['step']} (tools: {done}).")
    return st


def _loop(st, max_iterations, token_limit, time_limit_seconds, crash_after_tools=None):
    start = time.monotonic()
    while True:
        if time.monotonic() - start >= time_limit_seconds:                       # 7c
            return _fallback(st, "FALLBACK_TIMEOUT", f"time limit of {time_limit_seconds}s reached")
        if st["llm_calls"] >= max_iterations:                                     # 7a
            return _fallback(st, "FALLBACK_MAX_ITERATIONS", f"{max_iterations} model calls without a final answer")
        resp = mock_llm.chat(st["messages"], tools=TOOLS_SPEC)
        st["llm_calls"] += 1
        st["tokens_used"] += resp.usage.total_tokens
        if st["tokens_used"] > token_limit:                                       # 7b
            return _fallback(st, "FALLBACK_TOKEN_LIMIT", f"token limit of {token_limit} exceeded")
        msg = resp.choices[0].message
        if not msg.tool_calls:
            st["final_answer"] = msg.content
            st["status"] = msg.content.split(":")[0]
            return st
        tc = msg.tool_calls[0]
        st["messages"].append({"role": "assistant", "content": None,
                               "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}]})
        result = TOOL_MAP[tc.name](**tc.arguments)
        st["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                               "content": json.dumps(result)})
        st["step"] += 1
        st["tools_executed"].append(tc.name)
        st["steps"].append({"step": st["step"], "tool": tc.name, "arguments": tc.arguments, "observation": result})
        save_checkpoint(st)
        if crash_after_tools is not None and len(st["tools_executed"]) == crash_after_tools:
            raise SimulatedCrash(f"simulated crash after tool #{crash_after_tools} ({tc.name})")


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after_tools=None):
    st = {"run_id": run_id or f"{scenario_id}_{int(time.time() * 1000)}", "scenario_id": scenario_id,
          "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_message}],
          "tokens_used": 0, "llm_calls": 0, "step": 0, "tools_executed": [], "steps": [],
          "status": None, "final_answer": None,
          "limits": {"max_iterations": max_iterations, "token_limit": token_limit,
                     "time_limit_seconds": time_limit_seconds}}
    return _finish(_loop(st, max_iterations, token_limit, time_limit_seconds, crash_after_tools))


def resume_agent(run_id):
    st = load_checkpoint(run_id)
    lim = st["limits"]
    print(f"  resuming {run_id} from step {st['step']} (tools so far: {st['tools_executed']})")
    return _finish(_loop(st, lim["max_iterations"], lim["token_limit"], lim["time_limit_seconds"]))


def _finish(st):
    save_checkpoint(st)
    refund = next((s["observation"]["refund_inr"] for s in reversed(st["steps"])
                   if s["tool"] in ("issue_refund", "escalate_to_human")), 0)
    out = {"run_id": st["run_id"], "scenario_id": st["scenario_id"], "status": st["status"],
           "final_answer": st["final_answer"], "refund_inr": refund, "llm_calls": st["llm_calls"],
           "tokens_used": st["tokens_used"], "steps": st["steps"]}
    with open(RUNS, "a", encoding="utf-8") as f:
        f.write(json.dumps(out, default=str) + "\n")
    return out


if __name__ == "__main__":
    sc = pd.read_csv("data/raw/agent_scenarios.csv", dtype=str)
    print("=" * 70, "\n6. SEVEN SCENARIOS\n" + "=" * 70)
    rows = []
    for s in sc.itertuples():
        r = run_agent(s.message, s.scenario_id, run_id=s.scenario_id)
        expected = s.expected_outcome if s.expected_outcome != "FALLBACK" else "FALLBACK_MAX_ITERATIONS"
        ok = r["status"] == expected and abs(r["refund_inr"] - float(s.expected_refund_inr)) < 0.01
        rows.append({"scenario": s.scenario_id, "status": r["status"], "refund_inr": r["refund_inr"],
                     "expected": f"{s.expected_outcome} {s.expected_refund_inr}", "llm_calls": r["llm_calls"],
                     "tokens_used": r["tokens_used"], "tools": " > ".join(x["tool"] for x in r["steps"]),
                     "match": ok})
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    assert res.match.all() and res.loc[0, "llm_calls"] == 5

    print("\n" + "=" * 70, "\n7. STOP SAFELY\n" + "=" * 70)
    s1 = sc.iloc[0]
    tok = run_agent(s1.message, "S1", token_limit=200, run_id="S1_token_limit_200")
    print("S1 with token_limit=200 ->", tok["status"], "| llm_calls", tok["llm_calls"],
          "| tokens", tok["tokens_used"], "| tools run", [x["tool"] for x in tok["steps"]])
    print("  final_answer:", tok["final_answer"])
    tmo = run_agent(s1.message, "S1", time_limit_seconds=0, run_id="S1_time_limit_0")
    print("S1 with time_limit_seconds=0 ->", tmo["status"], "| llm_calls", tmo["llm_calls"])
    print("  final_answer:", tmo["final_answer"])
    assert tok["status"] == "FALLBACK_TOKEN_LIMIT" and tmo["status"] == "FALLBACK_TIMEOUT"

    print("\n" + "=" * 70, "\n8. CRASH AND RESUME (S1)\n" + "=" * 70)
    rid = "S1_crash_resume"
    try:
        run_agent(s1.message, "S1", run_id=rid, crash_after_tools=2)
    except SimulatedCrash as e:
        before = load_checkpoint(rid)["tools_executed"]
        print("  crash:", e, "| tools executed before crash:", before)
    final = resume_agent(rid)
    tools = [x["tool"] for x in final["steps"]]
    print("  tools executed across crash + resume:", tools)
    print("  final status:", final["status"], "| refund:", final["refund_inr"], "| answer:", final["final_answer"])
    assert tools == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    assert final["status"] == "REFUND" and abs(final["refund_inr"] - 128.25) < 0.01

    print("\nledger lines:", sum(1 for _ in open(LEDGER)), "| agent run lines:", sum(1 for _ in open(RUNS)),
          "| checkpoints:", sorted(os.listdir("state")))
