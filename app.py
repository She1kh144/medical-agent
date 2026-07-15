import os
import json
from dotenv import load_dotenv
from typing import cast
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolUnionParam, ChatCompletionMessageFunctionToolCall

def search_medical_knowledge(query: str) -> str:
    return (
        "Статус поиска: медицинские документы пока не подключены. "
        f"Получен запрос: {query}."
    )

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
                "«В базе знаний недостаточно информации для ответа.»"
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
                "name": "search_medical_knowledge",
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

    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=messages,
        tools=tools,
        temperature=0,
        max_tokens=200,
        extra_body={"thinking": {"type": "disabled"}},
    )

    message = response.choices[0].message

    print("Text:", message.content)

    if message.tool_calls:
        tool_call = cast(ChatCompletionMessageFunctionToolCall, message.tool_calls[0])

        print("Tool name:", tool_call.function.name)
        print("Arguments:", tool_call.function.arguments)

        arguments = json.loads(tool_call.function.arguments)
        query = arguments["query"]

        tool_result = search_medical_knowledge(query)
        print("Tool result:", tool_result)

        # model_dump returns a dict, cast to the expected ChatCompletionMessageParam for typing
        messages.append(cast(ChatCompletionMessageParam, message.model_dump(exclude_none=True)))

        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_result,
            }
        )

        final_response = client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=messages,
            temperature=0,
            max_tokens=200,
            extra_body={"thinking": {"type": "disabled"}},
        )

        final_message = final_response.choices[0].message
        print("Final answer:", final_message.content)
    else:
        print("The model did not request a tool.")

if __name__ == "__main__":
    main()