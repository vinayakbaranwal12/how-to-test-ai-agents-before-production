"""Check rules that must hold on every run, whatever path the agent took.

A trace is a JSON list of steps. Each step has a tool name, its arguments,
a token count, and whether the call matched its schema.

Usage:
  python3 rules_check.py trace.json
"""
import json
import sys

ALLOWED_TOOLS = {"lookup_order", "search_kb", "draft_reply"}
FORBIDDEN_ARGS = {"delete", "refund"}
MAX_STEPS = 12
MAX_TOKENS = 40000


def check_rules(trace):
    violations = []
    if len(trace) > MAX_STEPS:
        violations.append(f"step budget: {len(trace)} steps, limit {MAX_STEPS}")
    total = sum(step["tokens"] for step in trace)
    if total > MAX_TOKENS:
        violations.append(f"token budget: {total} tokens, limit {MAX_TOKENS}")
    for i, step in enumerate(trace, start=1):
        if step["tool"] not in ALLOWED_TOOLS:
            violations.append(f"step {i}: tool not on the allow list: {step['tool']}")
        if not step["schema_ok"]:
            violations.append(f"step {i}: arguments did not match the tool schema")
        if FORBIDDEN_ARGS & set(step["args"]):
            violations.append(f"step {i}: forbidden argument in {sorted(FORBIDDEN_ARGS & set(step['args']))}")
    return violations


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        found = check_rules(json.load(f))
    print(f"{len(found)} violations")
    for v in found:
        print(v)
    sys.exit(1 if found else 0)
