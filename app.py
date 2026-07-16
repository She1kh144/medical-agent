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
                "Перед ответом на вопросы о лекарствах используй инструмент поиска. "
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
            "content": "Найди информацию об ибупрофене.",
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
        }
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

            print("Tool name:", tool_name)
            print("Arguments:", raw_arguments)

            if tool_name == "medical_rag_search":
                arguments = json.loads(raw_arguments)
                query = arguments["query"]
                tool_result = medical_rag_search(query)
            else:
                tool_result = (
                    f"Error: Unknown tool '{tool_name}' called."
                )

            print("Tool result:", tool_result)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                }
            )

    raise RuntimeError(
        "Agent did not complete its task within the allowed number of steps."
    )

if __name__ == "__main__":
    main()