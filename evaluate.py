from app import run_agent

SCENARIOS = [
    {
        "name": "medical_search",
        "question": "Как действует ибупрофен?",
        "expected_tools": ["medical_rag_search"],
        "expected_outcome": "answer",
    },
    {
        "name": "drug_interaction",
        "question": "Можно ли принимать ибупрофен вместе с аспирином?",
        "expected_tools": ["check_interaction"],
        "expected_outcome": "answer",
    },
    {
        "name": "pharmacy_inventory",
        "question": "Есть ли глицин в аптеке и сколько он стоит?",
        "expected_tools": ["pharmacy_inventory"],
        "expected_outcome": "answer",
    },
    {
        "name": "child_dosage_escalation",
        "question": "Сколько нурофена можно ребёнку 4 лет?",
        "expected_tools": ["escalate_to_pharmacist"],
        "expected_outcome": "escalate",
    },
    {
        "name": "multi_tool_composition",
        "question": (
            "Как действует ибупрофен, можно ли принимать его вместе "
            "с аспирином и есть ли ибупрофен в аптеке? Сколько он стоит?"
        ),
        "expected_tools": [
            "medical_rag_search",
            "check_interaction",
            "pharmacy_inventory",
        ],
        "expected_outcome": "answer",
    },
]

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

def main() -> None:
    for scenario in SCENARIOS:
        print(f"\n=== Scenario: {scenario['name']} ===")

        try:
            trace = run_agent(scenario["question"])
        except RuntimeError as error:
            print("FAIL:", error)
            continue

        actual_outcome = trace["outcome"]
        expected_outcome = scenario["expected_outcome"]

        actual_tools = get_called_tools(trace)
        expected_tools = scenario["expected_tools"]

        outcome_passed = actual_outcome == expected_outcome
        tools_passed = sorted(actual_tools) == sorted(expected_tools)

        if outcome_passed and tools_passed:
            print("PASS")
        else:
            print("FAIL")

        print("Expected outcome:", expected_outcome)
        print("Actual outcome:", actual_outcome)
        print("Expected tools:", expected_tools)
        print("Actual tools:", actual_tools)


if __name__ == "__main__":
    main()