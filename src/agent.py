from openai import OpenAI
from dotenv import load_dotenv
import json
from tools import check_order_status, escalate_to_human

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
    }
]

# Step 2 — Map tool names to actual Python functions
TOOL_MAP = {
    "check_order_status": check_order_status,
    "escalate_to_human": escalate_to_human
}

def run_agent(user_message: str) -> str:
    messages = [
        {"role": "system", "content": """You are a Zomato customer support agent.
Use tools when needed to help customers.
Always be helpful and precise."""},
        {"role": "user", "content": user_message}
    ]

    # Step 3 — First LLM call: should it use a tool or answer directly?
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=TOOLS
    )

    message = response.choices[0].message

    # Step 4 — Did the LLM decide to use a tool?
    if message.tool_calls:
        tool_call = message.tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)

        print(f"Agent using tool: {tool_name} with args: {tool_args}")

        # Step 5 — Actually run the tool
        tool_result = TOOL_MAP[tool_name](**tool_args)

        # Step 6 — Give the result back to the LLM
        messages.append(message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": tool_result
        })

        # Step 7 — Second LLM call: now generate final answer
        final_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS
        )
        return final_response.choices[0].message.content

    # No tool needed — LLM answered directly
    return message.content


# Test it
if __name__ == "__main__":
    print(run_agent("What is the status of my order #67890?"))
    print("---")
    # Edit the test at the bottom of agent.py to this:
    print(run_agent("I have been waiting 3 hours and nobody is helping me, I want to speak to a manager NOW"))