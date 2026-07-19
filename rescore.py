import json

from collections import defaultdict

from evaluate import (
    load_scenarios,
    get_called_tools,
    get_assistant_text,
    classify_outcome,
    run_checks,
)

def load_latest_traces(path: str = "traces/runs.jsonl") -> dict[str, dict]:
    """Returns the newest trace per run_label."""
    latest: dict[str, dict] = {}

    with open(path, encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            label = record.get("run_label")

            if label:
                latest[label] = record

    return latest

def main() -> None:
    scenarios = load_scenarios()
    traces = load_latest_traces()

    results: list[tuple[dict, bool]] = []

    for scenario in scenarios:
        trace = traces.get(scenario["id"])

        if trace is None:
            print(f"{scenario['id']}: no trace found, skipping")
            continue

        outcome = str(trace["outcome"])
        called_tools = get_called_tools(trace)
        assistant_text = get_assistant_text(trace)

        outcome = classify_outcome(outcome, called_tools, assistant_text)

        failures = run_checks(scenario, called_tools, outcome, assistant_text)
        passed = len(failures) == 0

        results.append((scenario, passed))

        if not passed:
            print(f"\n=== {scenario['id']} [{scenario['category']}] ===")
            for failure in failures:
                print("  -", failure)

        if scenario["category"] == "clarify":
            print(
                f"\n[clarify diagnostics] {scenario['id']}: "
                f"tools={called_tools}, has_question={'?' in assistant_text}"
            )

    by_category: dict[str, list[bool]] = defaultdict(list)

    for scenario, passed in results:
        by_category[scenario["category"]].append(passed)

    print("\n=== Summary (rescored) ===")
    for category, flags in sorted(by_category.items()):
        print(f"{category:15} {sum(flags):2}/{len(flags)}")

    total = [passed for _, passed in results]
    print(f"\nTotal: {sum(total)}/{len(total)}")

if __name__ == "__main__":
    main()