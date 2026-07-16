import os
import json
import requests
from dotenv import load_dotenv
from typing import cast
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolUnionParam, ChatCompletionMessageFunctionToolCall

def medical_rag_search(query: str) -> str:
    response = requests.get(
        "http://localhost:8000/search",
        params={"query": query, "k": 5},
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()

    return json.dumps(data, ensure_ascii=False)

def check_interaction(drug_a: str, drug_b: str) -> str:
    query = (
        f"Взаимодействие препаратов {drug_a} и {drug_b}. "
        "Совместное применение, противопоказания и возможные риски."
    )

    return medical_rag_search(query)

def pharmacy_inventory(drug: str) -> str:
    base_url = "http://localhost:8001"

    stock_response = requests.get(
        f"{base_url}/stock/{drug}",
        timeout=10,
    )
    stock_response.raise_for_status()

    price_response = requests.get(
        f"{base_url}/price/{drug}",
        timeout=10,
    )
    price_response.raise_for_status()

    result = {
        "stock": stock_response.json(),
        "price": price_response.json(),
    }

    return json.dumps(result, ensure_ascii=False)

def escalate_to_pharmacist(reason: str) -> str:
    result = {
        "outcome": "escalate",
        "reason": reason,
    }

    return json.dumps(result, ensure_ascii=False)

def main() -> None:
    # Load environment variables from .env file
    load_dotenv()

    client = OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )

    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": (
                "Ты — помощник по медицинской информации. "
                "Перед ответом выбери подходящий из доступных инструментов. "
                "Отвечай только на основании медицинских фактов, содержащихся "
                "в результате инструмента. Не используй свои внутренние знания. "   
                "Если результат не содержит медицинских фактов, ответь точно: "
                "'В предоставленных документах нет ответа на этот вопрос'"

                "Не придумывай информацию, отсутствующую в контексте. "
                "В конце ответа укажи источник из предоставленного контекста (не придумывай источники). "
                "Всегда добавляй: 'Это не медицинская консультация, обратитесь к врачу.'"
            ),
        },
        {
            "role": "user",
            "content": "Сколько нурофена можно ребёнку 4 лет?",
        },
    ]

    tools: list[ChatCompletionToolUnionParam] = [
        {
            "type": "function",
            "function": {
                "name": "medical_rag_search",
                "description": (
                    "Ищет информацию о лекарственных препаратах "
                    "в медицинской базе знаний."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Поисковый запрос о лекарственном препарате."
                            ),
                        }
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "check_interaction",
                "description": (
                    "Проверяет известные взаимодействия между двумя "
                    "лекарственными препаратами по медицинской базе знаний. "
                    "Используй этот инструмент, когда пользователь спрашивает "
                    "о совместном применении двух препаратов."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "drug_a": {
                            "type": "string",
                            "description": "Название первого препарата.",
                        },
                        "drug_b": {
                            "type": "string",
                            "description": "Название второго препарата.",
                        },
                    },
                    "required": ["drug_a", "drug_b"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "pharmacy_inventory",
                "description": (
                    "Проверяет наличие и цену лекарственного препарата "
                    "в аптеке. Используй этот инструмент, когда пользователь "
                    "спрашивает, есть ли препарат в наличии, сколько единиц "
                    "осталось или сколько он стоит."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "drug_name": {
                            "type": "string",
                            "description": "Название лекарственного препарата.",
                        }
                    },
                    "required": ["drug_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "escalate_to_pharmacist",
                "description": (
                    "Передает запрос фармацевту и завершает работу агента. "
                    "Используй этот инструмент, если пользователь просит "
                    "подобрать дозировку лекарства для ребенка или запрос "
                    "требует индивидуального решения медицинского специалиста."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": (
                                "Краткая причина передачи запроса фармацевту."
                            ),
                        }
                    },
                    "required": ["reason"],
                },
            },
        },
    ]

    max_steps = 5

    for step in range(1, max_steps + 1):
        print(f"\n--- Agent's step {step} ---")

        response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=messages,
            tools=tools,
            temperature=0,
            max_tokens=500,
            extra_body={"thinking": {"type": "disabled"}},
        )

        message = response.choices[0].message
        messages.append(cast(ChatCompletionMessageParam, message.model_dump(exclude_none=True))) # To satisfy type checker

        if not message.tool_calls:
            print("Final answer:", message.content)
            return

        for tool_call in message.tool_calls:
            tool_call = cast(ChatCompletionMessageFunctionToolCall, tool_call)

            tool_name = tool_call.function.name
            raw_arguments = tool_call.function.arguments

            arguments = json.loads(raw_arguments)

            print("Tool name:", tool_name)
            print("Arguments:", raw_arguments)

            if tool_name == "medical_rag_search":
                tool_result = medical_rag_search(arguments["query"])
            elif tool_name == "check_interaction":
                tool_result = check_interaction(arguments["drug_a"], arguments["drug_b"])
            elif tool_name == "pharmacy_inventory":
                tool_result = pharmacy_inventory(arguments["drug_name"])
            elif tool_name == "escalate_to_pharmacist":
                tool_result = escalate_to_pharmacist(arguments["reason"])
            else:
                tool_result = f"Error: Unknown tool '{tool_name}' called."

            #print("Tool result:", tool_result) # Prints raw chunks

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                }
            )

            if tool_name == "escalate_to_pharmacist":
                print("Escalation:", tool_result)
                return

    raise RuntimeError("Agent did not complete its task within the allowed number of steps.")

if __name__ == "__main__":
    main()