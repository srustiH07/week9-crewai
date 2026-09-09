from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
import os

# --------------------------------------------------
# 1. CONFIGURE LOCAL OLLAMA LLM
# --------------------------------------------------

os.environ["OTEL_SDK_DISABLED"] = "true"

llm = LLM(
    model="ollama/qwen2.5:3b",
    base_url="http://localhost:11434"
)

# --------------------------------------------------
# 2. CONFIGURE WEB SEARCH TOOL
# --------------------------------------------------

search_tool = SerperDevTool()

# --------------------------------------------------
# 3. DEFINE AGENTS
# --------------------------------------------------

researcher = Agent(
    role="Researcher",
    goal=(
        "Research the given topic using reliable and current web information "
        "and identify accurate, useful findings."
    ),
    backstory=(
        "You are an experienced research analyst who searches the web, "
        "checks relevant information, and provides useful findings clearly."
    ),
    tools=[search_tool],
    llm=llm,
    verbose=True
)

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

# --------------------------------------------------
# 4. DEFINE TASKS
# --------------------------------------------------

research_task = Task(
    description=(
        "Research the topic: 'Applications of Artificial Intelligence "
        "in Education'. Use the web search tool to find current and "
        "reliable information. Identify important applications, benefits, "
        "challenges and practical examples. Include useful information "
        "from web results for the Writer."
    ),
    expected_output=(
        "Structured research notes based on current web information, "
        "covering AI applications in education, benefits, challenges "
        "and practical examples."
    ),
    agent=researcher
)

writing_task = Task(
    description=(
        "Using the web-based research provided by the Researcher, write "
        "a clear article about Applications of Artificial Intelligence "
        "in Education. Use suitable headings and simple explanations."
    ),
    expected_output=(
        "A well-structured article about AI applications in education "
        "based on the web research."
    ),
    agent=writer,
    context=[research_task]
)

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

# --------------------------------------------------
# 5. CREATE CREW
# --------------------------------------------------

crew = Crew(
    agents=[researcher, writer, reviewer],
    tasks=[research_task, writing_task, review_task],
    process=Process.sequential,
    verbose=True
)

# --------------------------------------------------
# 6. RUN CREW
# --------------------------------------------------

if __name__ == "__main__":

    print("\nStarting W9D2 CrewAI crew with web search...\n")

    if not os.getenv("SERPER_API_KEY"):
        raise EnvironmentError(
            "SERPER_API_KEY is not configured. "
            "Please configure the Serper API key before running."
        )

    result = crew.kickoff()

    print("\n" + "=" * 70)
    print("FINAL CREW OUTPUT - WITH WEB SEARCH")
    print("=" * 70)
    print(result)

    # --------------------------------------------------
    # 7. SAVE OUTPUT
    # --------------------------------------------------

    os.makedirs("output", exist_ok=True)

    output_file = os.path.join(
        "output",
        "crew_output_with_web.txt"
    )

    with open(output_file, "w", encoding="utf-8") as file:
        file.write(str(result))

    print(f"\nOutput saved to: {output_file}")