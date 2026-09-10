from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain.agents import create_agent


# Create the local Ollama chat model.
# ChatOllama supports tool calling through bind_tools().
llm = ChatOllama(
    model="qwen2.5:3b",
    base_url="http://localhost:11434"
)


# Calculator tool.
@tool
def calculator(expression: str) -> str:
    """Calculate a basic mathematical expression."""

    try:
        # Allow only basic mathematical characters.
        allowed_characters = "0123456789+-*/(). "

        if not all(
            character in allowed_characters
            for character in expression
        ):
            return "Invalid mathematical expression."

        result = eval(
            expression,
            {"__builtins__": {}},
            {}
        )

        return str(result)

    except Exception:
        return "Unable to calculate the expression."


# Web search stub.
@tool
def web_search_stub(query: str) -> str:
    """Simulate a web search and return a sample result."""

    sample_results = {
        "artificial intelligence":
            "Artificial Intelligence enables computers to perform "
            "tasks that normally require human intelligence.",

        "machine learning":
            "Machine Learning allows systems to learn patterns "
            "from data and make predictions or decisions.",

        "langchain":
            "LangChain is a framework for building applications "
            "powered by language models and tools."
    }

    query_lower = query.lower()

    for keyword, result in sample_results.items():
        if keyword in query_lower:
            return result

    return (
        f"Web search stub result for '{query}': "
        "No specific result was found in the sample search database."
    )


# Create the two-tool agent.
agent = create_agent(
    model=llm,
    tools=[
        calculator,
        web_search_stub
    ],
    system_prompt=(
        "You are a helpful assistant with two tools: "
        "a calculator and a web search stub. "
        "Use the calculator for mathematical calculations. "
        "Use the web search stub when the user asks for "
        "information that requires a web search. "
        "Give clear and concise answers."
    )
)


# Three tasks required by W9D4.
tasks = [
    "Calculate 125 * 8 + 50.",
    "Search for information about Artificial Intelligence.",
    "Calculate (250 / 5) + 75."
]


# Store results for the evidence file.
results = []

print("\nW9D4 - Two-Tool Agent Test")
print("=" * 60)


for number, task in enumerate(tasks, start=1):

    print(f"\nTask {number}: {task}")
    print("-" * 60)

    response = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": task
                }
            ]
        }
    )

    # Get the final assistant response.
    final_message = response["messages"][-1].content

    print(f"Agent: {final_message}")

    results.append(
        f"TASK {number}: {task}\n"
        f"AGENT: {final_message}\n"
        f"{'-' * 60}\n"
    )


# Save the results for evidence.
output_file = (
    "day4-hybrid-agent-systems/output/agent_output.txt"
)

with open(output_file, "w", encoding="utf-8") as file:

    file.write(
        "W9D4 - Two-Tool Agent Output\n\n"
    )

    file.write("\n".join(results))


print("\n" + "=" * 60)
print(f"Output saved to: {output_file}")