from crewai import Agent, Task, Crew, Process, LLM
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
# 2. DEFINE AGENTS
# --------------------------------------------------

researcher = Agent(
    role="Researcher",
    goal=(
        "Research the given topic using your available knowledge "
        "and identify accurate and useful information."
    ),
    backstory=(
        "You are an experienced research analyst who collects "
        "important information about a topic and presents it clearly."
    ),
    llm=llm,
    verbose=True
)

writer = Agent(
    role="Writer",
    goal=(
        "Transform the research findings into a clear and "
        "well-structured article."
    ),
    backstory=(
        "You are a professional technical writer who converts "
        "research findings into simple, readable and informative content."
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
        "Research the topic 'Applications of Artificial Intelligence "
        "in Healthcare'. Identify important applications, benefits, "
        "challenges and real-world uses using your available knowledge."
    ),
    expected_output=(
        "Structured research notes covering AI applications in healthcare, "
        "benefits, challenges and examples."
    ),
    agent=researcher
)

writing_task = Task(
    description=(
        "Using the research findings from the Researcher, write a clear "
        "and informative article about Applications of Artificial "
        "Intelligence in Healthcare. Use suitable headings."
    ),
    expected_output=(
        "A well-structured article explaining AI applications in healthcare."
    ),
    agent=writer,
    context=[research_task]
)

review_task = Task(
    description=(
        "Review the article produced by the Writer. Check accuracy, "
        "clarity, completeness, structure and usefulness. "
        "Identify weaknesses and provide improvement suggestions."
    ),
    expected_output=(
        "A review containing quality assessment, identified weaknesses "
        "and specific improvement suggestions."
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

    print("\nStarting W9D3 Research Crew WITHOUT web search...\n")

    result = crew.kickoff()

    print("\n" + "=" * 70)
    print("W9D3 OUTPUT - WITHOUT WEB SEARCH")
    print("=" * 70)
    print(result)

    # --------------------------------------------------
    # 6. SAVE OUTPUT
    # --------------------------------------------------

    os.makedirs("output", exist_ok=True)

    output_file = os.path.join(
        "output",
        "crew_output_without_web.txt"
    )

    with open(output_file, "w", encoding="utf-8") as file:
        file.write(str(result))

    print(f"\nOutput saved to: {output_file}")