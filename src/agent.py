from openai import OpenAI
from dotenv import load_dotenv
import json
from tools import check_order_status, escalate_to_human
import sys
sys.path.append('..')
from search import search


load_dotenv()
client = OpenAI()

# Step 1 — Tell the LLM what tools exist and how to use them
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_order_status",
            "description": "Check the current status of a customer order",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID to check"
                    }
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Escalate a customer issue to a human agent when unable to resolve",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "The reason why escalation is needed"
                    }
                },
                "required": ["reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "Search Zomato policy documents to answer customer questions about refunds, delivery, and payments",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The customer's question to search for"
                    }
                },
                "required": ["query"]
            }
        }
    }

]

# Step 2 — Map tool names to actual Python functions
TOOL_MAP = {
    "check_order_status": check_order_status,
    "escalate_to_human": escalate_to_human,
    "search_policy": lambda query: "\n\n".join(search(query))
}

def run_agent(user_message: str) -> str:
    messages = [
        {"role": "system", "content": """You are a Zomato customer support agent.
Use tools when needed to help customers.
Always be helpful and precise."""},
        {"role": "user", "content": user_message}
    ]

    # Keep looping until LLM stops calling tools
    while True:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS
        )

        message = response.choices[0].message

        # No tool call — LLM is done, return final answer
        if not message.tool_calls:
            return message.content

        # Tool call requested — run it
        tool_call = message.tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)

        print(f"Agent using tool: {tool_name} with args: {tool_args}")

        tool_result = TOOL_MAP[tool_name](**tool_args)

        print(f"Tool result: {tool_result}")

        # Add tool call + result back into conversation
        messages.append(message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": tool_result
        })
        # Loop continues — LLM decides next step


# Test it
if __name__ == "__main__":
    print(run_agent("What is the refund policy for cancelled orders?"))
    print("---")
    print(run_agent("Check my order #67890"))