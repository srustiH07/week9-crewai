from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
import os

# Disable OpenTelemetry tracing for local execution.
os.environ["OTEL_SDK_DISABLED"] = "true"

# Configure the local Ollama LLM.
llm = LLM(
    model="ollama/qwen2.5:3b",
    base_url="http://localhost:11434"
)

# Create the web search tool.
search_tool = SerperDevTool()

# Researcher agent searches for current information.
researcher = Agent(
    role="Researcher",
    goal=(
        "Research the given topic using reliable and current web information "
        "and identify accurate, useful findings."
    ),
    backstory=(
        "You are an experienced research analyst who searches the web, "
        "checks relevant information and presents useful findings clearly."
    ),
    tools=[search_tool],
    llm=llm,
    verbose=True
)

# Writer agent converts research into an article.
writer = Agent(
    role="Writer",
    goal=(
        "Turn research findings into a clear, informative and "
        "well-structured article."
    ),
    backstory=(
        "You are a professional technical writer who transforms "
        "research findings into readable and informative content."
    ),
    llm=llm,
    verbose=True
)

# Reviewer agent checks the final article.
reviewer = Agent(
    role="Reviewer",
    goal=(
        "Review the article for accuracy, clarity, completeness "
        "and overall quality."
    ),
    backstory=(
        "You are a strict content reviewer who checks factual accuracy, "
        "logical flow, clarity and usefulness."
    ),
    llm=llm,
    verbose=True
)

# Research task using real web search.
research_task = Task(
    description=(
        "Research the topic 'Applications of Artificial Intelligence "
        "in Healthcare'. Use the web search tool to find current and "
        "reliable information. Identify important applications, benefits, "
        "challenges and real-world examples."
    ),
    expected_output=(
        "Structured research notes based on current web information "
        "covering AI applications in healthcare, benefits, challenges "
        "and real-world examples."
    ),
    agent=researcher
)

# Writer receives the Researcher's findings.
writing_task = Task(
    description=(
        "Using the web-based research provided by the Researcher, write "
        "a clear and informative article about Applications of Artificial "
        "Intelligence in Healthcare. Use suitable headings and simple "
        "explanations."
    ),
    expected_output=(
        "A well-structured article about AI applications in healthcare "
        "based on the web research."
    ),
    agent=writer,
    context=[research_task]
)

# Reviewer receives the Writer's article.
review_task = Task(
    description=(
        "Review the article produced by the Writer. Check accuracy, "
        "clarity, completeness, structure and usefulness. Identify "
        "weaknesses and provide specific improvement suggestions."
    ),
    expected_output=(
        "A review containing quality assessment, identified weaknesses "
        "and specific improvement suggestions."
    ),
    agent=reviewer,
    context=[writing_task]
)

# Create the sequential multi-agent crew.
crew = Crew(
    agents=[researcher, writer, reviewer],
    tasks=[research_task, writing_task, review_task],
    process=Process.sequential,
    verbose=True
)

if __name__ == "__main__":

    # Check that the Serper API key is configured.
    if not os.getenv("SERPER_API_KEY"):
        raise EnvironmentError(
            "SERPER_API_KEY is not configured. "
            "Please configure the Serper API key before running."
        )

    print("\nStarting W9D3 Research Crew WITH web search...\n")

    # Run the complete Researcher -> Writer -> Reviewer pipeline.
    result = crew.kickoff()

    print("\n" + "=" * 70)
    print("W9D3 OUTPUT - WITH WEB SEARCH")
    print("=" * 70)
    print(result)

    # Save the final result.
    os.makedirs("output", exist_ok=True)

    output_file = os.path.join(
        "output",
        "crew_output_with_web.txt"
    )

    with open(output_file, "w", encoding="utf-8") as file:
        file.write(str(result))

    print(f"\nOutput saved to: {output_file}")