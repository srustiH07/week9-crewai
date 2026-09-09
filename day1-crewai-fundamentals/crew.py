from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
import os

# --------------------------------------------------
# 1. CONFIGURE LOCAL OLLAMA LLM
# --------------------------------------------------

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
        "Research the given topic using reliable and up-to-date web information "
        "and identify accurate, useful findings."
    ),
    backstory=(
        "You are an experienced research analyst who searches the web, "
        "checks relevant information, and presents useful findings clearly."
    ),
    tools=[search_tool],
    llm=llm,
    verbose=True
)

writer = Agent(
    role="Writer",
    goal="Turn research findings into a clear and well-structured article.",
    backstory=(
        "You are a professional technical writer who can transform research "
        "notes into concise, readable, and informative content."
    ),
    llm=llm,
    verbose=True
)

reviewer = Agent(
    role="Reviewer",
    goal="Review the written article for accuracy, clarity, completeness, and quality.",
    backstory=(
        "You are a strict content reviewer who checks factual accuracy, "
        "logical flow, clarity, and whether the article addresses the topic properly."
    ),
    llm=llm,
    verbose=True
)


# --------------------------------------------------
# 4. DEFINE TASKS
# --------------------------------------------------

research_task = Task(
    description=(
        "Research the topic: 'Applications of Artificial Intelligence in Healthcare'. "
        "Use the web search tool to find current and reliable information. "
        "Identify important applications, benefits, challenges, and real-world uses. "
        "Include useful facts from the web results for the writer."
    ),
    expected_output=(
        "A structured set of research notes based on current web information, "
        "covering AI applications in healthcare, benefits, challenges, and examples."
    ),
    agent=researcher
)

writing_task = Task(
    description=(
        "Using the web-based research provided by the Researcher, write an "
        "informative article about applications of Artificial Intelligence in "
        "Healthcare. Organize the article with suitable headings and keep it "
        "easy to understand."
    ),
    expected_output=(
        "A clear and well-structured article about AI applications in healthcare "
        "based on the research findings."
    ),
    agent=writer,
    context=[research_task]
)

review_task = Task(
    description=(
        "Review the article produced by the Writer. Check its accuracy, clarity, "
        "completeness, structure, and usefulness. Identify weaknesses and provide "
        "specific improvement suggestions."
    ),
    expected_output=(
        "A review containing an overall quality assessment, identified issues, "
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
    print("\nStarting CrewAI research crew with web search...\n")

    if not os.getenv("SERPER_API_KEY"):
        raise EnvironmentError(
            "SERPER_API_KEY is not set. Please configure your Serper API key "
            "before running the web-search version."
        )

    result = crew.kickoff()

    print("\n" + "=" * 70)
    print("FINAL CREW OUTPUT - WITH WEB SEARCH")
    print("=" * 70)
    print(result)

    os.makedirs("output", exist_ok=True)

    output_file = os.path.join(
        "output",
        "crew_output_with_web.txt"
    )

    with open(output_file, "w", encoding="utf-8") as file:
        file.write(str(result))

    print(f"\nOutput saved to: {output_file}")