from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser


# Create the Ollama language model.
llm = OllamaLLM(
    model="qwen2.5:3b",
    base_url="http://localhost:11434"
)


# Create a prompt template.
prompt = PromptTemplate(
    input_variables=["topic"],
    template=(
        "Explain the following topic in simple and clear terms "
        "for an engineering student:\n\n{topic}"
    )
)


# Create an output parser.
output_parser = StrOutputParser()


# Build the LangChain pipeline:
# PromptTemplate -> Ollama LLM -> OutputParser
chain = prompt | llm | output_parser


# Five test inputs required by W9D4.
test_inputs = [
    "Artificial Intelligence",
    "Machine Learning",
    "Large Language Models",
    "Retrieval Augmented Generation",
    "Multi-Agent Systems"
]


# Run the chain with five inputs.
results = []

print("\nW9D4 - LangChain Chain Test")
print("=" * 60)

for number, topic in enumerate(test_inputs, start=1):

    print(f"\nInput {number}: {topic}")
    print("-" * 60)

    response = chain.invoke({"topic": topic})

    print(response)

    results.append(
        f"INPUT {number}: {topic}\n"
        f"{response}\n"
        f"{'-' * 60}\n"
    )


# Save the results for evidence.
output_file = "day4-hybrid-agent-systems/output/chain_output.txt"

with open(output_file, "w", encoding="utf-8") as file:
    file.write(
        "W9D4 - LangChain Chain Output\n"
        "PromptTemplate -> Ollama LLM -> OutputParser\n\n"
    )
    file.write("\n".join(results))


print("\n" + "=" * 60)
print(f"Output saved to: {output_file}")