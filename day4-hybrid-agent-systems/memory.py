from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.memory import ConversationBufferMemory


# Create the local Ollama language model.
llm = OllamaLLM(
    model="qwen2.5:3b",
    base_url="http://localhost:11434"
)


# Create conversation memory.
memory = ConversationBufferMemory(
    return_messages=False
)


# Create the prompt template.
prompt = PromptTemplate(
    input_variables=["history", "input"],
    template=(
        "You are a helpful AI assistant.\n\n"
        "Conversation history:\n"
        "{history}\n\n"
        "Current user message:\n"
        "{input}\n\n"
        "Answer clearly and briefly."
    )
)


# Create the output parser.
output_parser = StrOutputParser()


# Build the chain:
# PromptTemplate -> Ollama LLM -> OutputParser
chain = prompt | llm | output_parser


# Five conversation turns required by W9D4.
conversation = [
    "My name is Srusti.",
    "I am studying Information Science Engineering.",
    "What course am I studying?",
    "What is my name?",
    "Can you summarize what you know about me from this conversation?"
]


# Store all outputs for the evidence file.
results = []

print("\nW9D4 - ConversationBufferMemory Test")
print("=" * 60)

for turn, user_input in enumerate(conversation, start=1):

    # Load the conversation history before generating the response.
    history = memory.load_memory_variables({})["history"]

    print(f"\nTurn {turn}")
    print(f"User: {user_input}")

    # Send the current input together with previous history.
    response = chain.invoke(
        {
            "history": history,
            "input": user_input
        }
    )

    print(f"AI: {response}")

    # Save the current interaction into memory.
    memory.save_context(
        {"input": user_input},
        {"output": response}
    )

    results.append(
        f"TURN {turn}\n"
        f"User: {user_input}\n"
        f"AI: {response}\n"
        f"{'-' * 60}\n"
    )


# Display the complete conversation history.
final_history = memory.load_memory_variables({})["history"]

print("\n" + "=" * 60)
print("COMPLETE CONVERSATION HISTORY")
print("=" * 60)
print(final_history)


# Save the results for evidence.
output_file = "day4-hybrid-agent-systems/output/memory_output.txt"

with open(output_file, "w", encoding="utf-8") as file:
    file.write("W9D4 - ConversationBufferMemory Output\n\n")
    file.write("\n".join(results))
    file.write("\n\nCOMPLETE CONVERSATION HISTORY\n")
    file.write("=" * 60)
    file.write("\n")
    file.write(final_history)


print("\n" + "=" * 60)
print(f"Output saved to: {output_file}")