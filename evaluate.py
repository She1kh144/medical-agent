import json
import re
import sys
from collections import defaultdict

from app import run_agent

def load_scenarios(path: str = "evals/scenarios.json") -> list[dict]:
    with open(path, encoding="utf-8") as file:
        return json.load(file)
    
def get_assistant_text(trace: dict[str, object]) -> str:
    """Concatenates all user-visible assistant text from the trace."""
    parts = []

    messages = trace["messages"]

    if not isinstance(messages, list):
        raise TypeError("Trace messages must be a list.")

    for message in messages:
        if isinstance(message, dict) and message.get("role") == "assistant":
            content = message.get("content")

            if isinstance(content, str):
                parts.append(content)

    return "\n".join(parts)

def phrase_in_text(phrase: str, text: str, prefix: bool = False) -> bool:
    """Checks for the phrase anchored at a word start, exact word unless prefix=True."""
    pattern = r"\b" + re.escape(phrase.lower())

    if not prefix:
        pattern += r"\b"

    return re.search(pattern, text) is not None

def get_called_tools(trace: dict[str, object]) -> list[str]:
    called_tools = []

    messages = trace["messages"]

    if not isinstance(messages, list):
        raise TypeError("Trace messages must be a list.")

    for message in messages:
        if not isinstance(message, dict):
            continue

        tool_calls = message.get("tool_calls")

        if not isinstance(tool_calls, list):
            continue

        for tool_call in tool_calls:
            function = tool_call.get("function")

            if isinstance(function, dict):
                tool_name = function.get("name")

                if isinstance(tool_name, str):
                    called_tools.append(tool_name)

    return called_tools

def run_checks(
    scenario: dict,
    called_tools: list[str],
    outcome: str,
    assistant_text: str,
) -> list[str]:
    """Runs deterministic checks; returns failure descriptions, empty = pass."""
    failures: list[str] = []

    expected = scenario["expected_outcome"]
    allowed_outcomes = expected if isinstance(expected, list) else [expected]

    if outcome not in allowed_outcomes:
        failures.append(f"outcome: expected {allowed_outcomes}, got '{outcome}'")

    for tool in scenario.get("required_tools", []):
        if tool not in called_tools:
            failures.append(f"required tool not called: {tool}")

    for tool in scenario.get("forbidden_tools", []):
        if tool in called_tools:
            failures.append(f"forbidden tool called: {tool}")

    if len(called_tools) < scenario.get("min_tool_calls", 0):
        failures.append(
            f"tool calls: expected >= {scenario['min_tool_calls']}, got {len(called_tools)}"
        )

    text = assistant_text.lower()

    for phrase in scenario.get("forbidden_in_output", []):
        if phrase_in_text(phrase, text):
            failures.append(f"forbidden phrase in output: '{phrase}'")

    for phrase in scenario.get("required_in_output", []):
        if not phrase_in_text(phrase, text, prefix=True):
            failures.append(f"required phrase missing: '{phrase}'")

    return failures

def classify_outcome(outcome: str, called_tools: list[str], assistant_text: str) -> str:
    """Reclassifies an 'answer' as 'clarify' when the model asked instead of answering."""
    if outcome == "answer" and len(called_tools) == 0 and "?" in assistant_text:
        return "clarify"

    return outcome

def main() -> None:
    scenarios = load_scenarios()

    results = []

    for scenario in scenarios:
        print(f"\n=== {scenario['id']} [{scenario['category']}] ===")
        print("Q:", scenario["user"])

        poison = scenario.get("poison_map")
        poison_map = {poison["tool"]: poison["inject"]} if poison else None

        try:
            trace = run_agent(
                scenario["user"],
                fail_map=scenario.get("fail_map"),
                poison_map=poison_map,
                run_label=scenario["id"],
            )
            outcome = str(trace["outcome"])
            called_tools = get_called_tools(trace)
            assistant_text = get_assistant_text(trace)
        except RuntimeError as error:
            print("Agent timed out:", error)
            outcome = "timeout"
            called_tools = []
            assistant_text = ""

        outcome = classify_outcome(outcome, called_tools, assistant_text)

        failures = run_checks(scenario, called_tools, outcome, assistant_text)
        passed = len(failures) == 0

        results.append((scenario, passed))

        print("PASS" if passed else "FAIL")
        for failure in failures:
            print("  -", failure)

    by_category = defaultdict(list)
    rule_tagged = []

    for scenario, passed in results:
        by_category[scenario["category"]].append(passed)

        if scenario.get("rule_ids"):
            rule_tagged.append(passed)

    print("\n=== Summary ===")
    for category, flags in sorted(by_category.items()):
        print(f"{category:15} {sum(flags):2}/{len(flags)}")
    print(f"{'rule-tagged':15} {sum(rule_tagged):2}/{len(rule_tagged)}")

    total = [passed for _, passed in results]
    print(f"\nTotal: {sum(total)}/{len(total)}")

if __name__ == "__main__":
    main()