import os
import sys
import json
from dotenv import load_dotenv
from openai import OpenAI

from evaluate import load_scenarios, get_assistant_text
from rescore import load_latest_traces

load_dotenv() 

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

JUDGE_SYSTEM_PROMPT = """Ты — строгий судья обоснованности ответов медицинского ассистента.

Тебе дают: ВОПРОС пользователя (QUESTION), СВИДЕТЕЛЬСТВА из инструментов (EVIDENCE) и итоговый ОТВЕТ ассистента (ANSWER).

Твоя единственная задача: найти в ОТВЕТЕ каждое конкретное фактическое утверждение, которое не подтверждается СВИДЕТЕЛЬСТВАМИ.

Правила:
- Конкретное утверждение — это любой конкретный медицинский факт: дозировки, временные интервалы, числа, механизмы действия, взаимодействия, эффекты, предупреждения, названия источников.
- Подтверждено — значит смысл утверждения присутствует в СВИДЕТЕЛЬСТВАХ. Дословное совпадение не требуется.
- Правдоподобие в реальном мире НЕ ИМЕЕТ ЗНАЧЕНИЯ. Утверждение, верное в реальной жизни, но отсутствующее в СВИДЕТЕЛЬСТВАХ, — не подтверждено. Это самая частая ошибка при оценке. Не допускай её.
- Никогда не считай утверждениями: стандартный дисклеймер, совет обратиться к врачу, отказы отвечать, сообщения о том, что в документах нет ответа.
- Приводи каждое неподтверждённое утверждение как короткую дословную цитату из ОТВЕТА.

Отвечай только JSON, без markdown-ограждений:
{"unsupported_claims": ["<дословная цитата>", "..."]}
Если все утверждения подтверждены, ответь: {"unsupported_claims": []}"""

def get_user_question(trace: dict) -> str:
    for message in trace["messages"]:
        if isinstance(message, dict) and message.get("role") == "user":
            return message["content"]

    raise ValueError("No user message in trace.")

def get_tool_evidence(trace: dict) -> str:
    parts = []

    for message in trace["messages"]:
        if isinstance(message, dict) and message.get("role") == "tool":
            parts.append(f"--- TOOL RESULT {len(parts) + 1} ---\n{message['content']}")

    return "\n\n".join(parts)

def judge_trace(trace: dict) -> dict:
    user_content = (
        f"QUESTION:\n{get_user_question(trace)}\n\n"
        f"EVIDENCE:\n{get_tool_evidence(trace)}\n\n"
        f"ANSWER:\n{get_assistant_text(trace)}"
    )

    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
        max_tokens=1000,
        extra_body={"thinking": {"type": "disabled"}},
    )

    content = response.choices[0].message.content or ""
    raw = content.strip()

    # Sometimes the model wraps JSON in code fences despite instructions
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw
        raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        claims = json.loads(raw)["unsupported_claims"]
    except (json.JSONDecodeError, KeyError):
        print("JUDGE RAW OUTPUT:", repr(response.choices[0].message.content))
        raise

    return {"grounded": len(claims) == 0, "unsupported_claims": claims}

def main() -> None:
    scenarios = load_scenarios()
    traces = load_latest_traces()

    graded = 0
    grounded_count = 0
    skipped = []

    for scenario in scenarios:
        if not scenario.get("judge_eligible"):
            continue

        label = scenario["id"]
        trace = traces.get(label)

        if trace is None or str(trace["outcome"]) != "answer":
            skipped.append(label)
            continue

        result = judge_trace(trace)
        graded += 1
        grounded_count += result["grounded"]

        print(f"{label}: grounded={result['grounded']}")

        for claim in result["unsupported_claims"]:
            print("   -", claim)

    print(f"\nGrounded: {grounded_count}/{graded}")
    print("Skipped (no answer to grade):", ", ".join(skipped) if skipped else "none")

if __name__ == "__main__":
    main()