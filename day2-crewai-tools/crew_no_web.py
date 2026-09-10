from crewai import Agent, Task, Crew, Process, LLM
import os

# --------------------------------------------------
# 1. CONFIGURE LOCAL OLLAMA LLM
# --------------------------------------------------

llm = LLM(
    model="ollama/qwen2.5:3b",
    base_url="http://localhost:11434"
)

# --------------------------------------------------
# 2. DEFINE AGENTS
# --------------------------------------------------

researcher = Agent(
    role="Researcher",
    goal=(
        "Research the given topic and identify accurate, useful "
        "and well-organized information."
    ),
    backstory=(
        "You are an experienced research analyst who collects "
        "important information and presents clear research findings."
    ),
    llm=llm,
    verbose=True
)

writer = Agent(
    role="Writer",
    goal=(
        "Convert research findings into a clear, informative "
        "and well-structured article."
    ),
    backstory=(
        "You are a professional technical writer who transforms "
        "research information into readable content."
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
        "You are a strict reviewer who identifies weaknesses, "
        "missing information and areas for improvement."
    ),
    llm=llm,
    verbose=True
)

# --------------------------------------------------
# 3. DEFINE TASKS
# --------------------------------------------------

research_task = Task(
    description=(
        "Research the topic: 'Applications of Artificial Intelligence "
        "in Education'. Identify important applications, benefits, "
        "challenges and practical examples. Provide useful research "
        "notes for the Writer."
    ),
    expected_output=(
        "Structured research notes covering AI applications in education, "
        "benefits, challenges and examples."
    ),
    agent=researcher
)

writing_task = Task(
    description=(
        "Using the research provided by the Researcher, write a clear "
        "article about Applications of Artificial Intelligence in Education. "
        "Use suitable headings and simple explanations."
    ),
    expected_output=(
        "A well-structured article explaining AI applications in education."
    ),
    agent=writer,
    context=[research_task]
)

review_task = Task(
    description=(
        "Review the article produced by the Writer. Check accuracy, "
        "clarity, completeness, structure and usefulness. Provide "
        "specific improvement suggestions."
    ),
    expected_output=(
        "A review containing quality assessment, weaknesses and "
        "specific improvement suggestions."
    ),
    agent=reviewer,
    context=[writing_task]
)

# --------------------------------------------------
# 4. CREATE CREW
# --------------------------------------------------

crew = Crew(
    agents=[researcher, writer, reviewer],
    tasks=[research_task, writing_task, review_task],
    process=Process.sequential,
    verbose=True
)

# --------------------------------------------------
# 5. RUN CREW
# --------------------------------------------------

if __name__ == "__main__":

    # Disable telemetry for a smoother local run
    os.environ["OTEL_SDK_DISABLED"] = "true"

    print("\nStarting W9D2 CrewAI crew without web search...\n")

    result = crew.kickoff()

    print("\n" + "=" * 70)
    print("FINAL CREW OUTPUT - WITHOUT WEB SEARCH")
    print("=" * 70)
    print(result)

    # --------------------------------------------------
    # 6. SAVE OUTPUT
    # --------------------------------------------------

    os.makedirs("output", exist_ok=True)

    output_file = os.path.join(
        "output",
        "crew_output_no_web.txt"
    )

    with open(output_file, "w", encoding="utf-8") as file:
        file.write(str(result))

    print(f"\nOutput saved to: {output_file}")