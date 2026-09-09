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
    goal="Research the given topic and identify accurate, useful information.",
    backstory=(
        "You are an experienced research analyst who collects reliable "
        "information, identifies important facts, and presents findings clearly."
    ),
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
    goal=(
        "Review the written article for accuracy, clarity, completeness, "
        "and quality."
    ),
    backstory=(
        "You are a strict content reviewer who checks factual accuracy, "
        "logical flow, clarity, and whether the article addresses the topic properly."
    ),
    llm=llm,
    verbose=True
)

# --------------------------------------------------
# 3. DEFINE TASKS
# --------------------------------------------------

research_task = Task(
    description=(
        "Research the topic: 'Applications of Artificial Intelligence in Healthcare'. "
        "Identify important applications, benefits, challenges, and real-world uses. "
        "Provide concise research notes for the writer."
    ),
    expected_output=(
        "A structured set of research notes covering AI applications in healthcare, "
        "benefits, challenges, and examples."
    ),
    agent=researcher
)

writing_task = Task(
    description=(
        "Using the research provided by the Researcher, write an informative article "
        "about applications of Artificial Intelligence in Healthcare. "
        "Organize the article with suitable headings and keep it easy to understand."
    ),
    expected_output=(
        "A clear and well-structured article about AI applications in healthcare."
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
    print("\nStarting CrewAI research crew with Ollama...\n")

    result = crew.kickoff()

    print("\n" + "=" * 70)
    print("FINAL CREW OUTPUT - WITHOUT WEB SEARCH")
    print("=" * 70)
    print(result)

    # Save output
    os.makedirs("output", exist_ok=True)

    output_file = os.path.join(
        "output",
        "crew_output_no_web.txt"
    )

    with open(output_file, "w", encoding="utf-8") as file:
        file.write(str(result))

    print(f"\nOutput saved to: {output_file}")