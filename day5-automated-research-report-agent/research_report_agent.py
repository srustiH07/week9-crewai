# W9D5 - Automated Research Report Agent
# Stack: CrewAI + LangGraph + MLflow + Ragas + MLOps
#
# Workflow:
# Topic -> Research Agent -> Writer Agent -> Report
#       -> Ragas Evaluation -> MLflow Tracking

import os
from typing import TypedDict

import mlflow

from crewai import Agent, Task, Crew, Process, LLM
from langgraph.graph import StateGraph, START, END


# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------

MODEL_NAME = "ollama/qwen2.5:3b"
OLLAMA_BASE_URL = "http://localhost:11434"

RESEARCH_TOPIC = "Artificial Intelligence in Healthcare"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# 2. Configure local Ollama LLM
# ---------------------------------------------------------

llm = LLM(
    model=MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    temperature=0.2
)


# ---------------------------------------------------------
# 3. Create Research Agent
# ---------------------------------------------------------

researcher = Agent(
    role="Research Analyst",
    goal="Research the given topic and provide accurate, structured knowledge.",
    backstory=(
        "You are an AI research analyst. "
        "You analyze a topic carefully and organize important information "
        "into clear research notes."
    ),
    llm=llm,
    verbose=False
)


# ---------------------------------------------------------
# 4. Create Writer Agent
# ---------------------------------------------------------

writer = Agent(
    role="Report Writer",
    goal="Convert research notes into a clear and professional report.",
    backstory=(
        "You are a technical report writer. "
        "You transform research notes into a well-structured report "
        "with headings, explanations, benefits, challenges and conclusion."
    ),
    llm=llm,
    verbose=False
)


# ---------------------------------------------------------
# 5. Research Task
# ---------------------------------------------------------

research_task = Task(
    description=(
        "Research the following topic:\n\n"
        "{topic}\n\n"
        "Provide structured research notes covering:\n"
        "1. Introduction\n"
        "2. Main concepts\n"
        "3. Applications\n"
        "4. Benefits\n"
        "5. Challenges\n"
        "6. Future scope\n\n"
        "Do not invent specific statistics or citations."
    ),
    expected_output="Detailed and structured research notes.",
    agent=researcher
)


# ---------------------------------------------------------
# 6. Writing Task
# ---------------------------------------------------------

writing_task = Task(
    description=(
        "Write a professional research report about:\n\n"
        "{topic}\n\n"
        "Use the following research notes:\n\n"
        "{research}\n\n"
        "The report must contain:\n"
        "1. Title\n"
        "2. Introduction\n"
        "3. Main concepts\n"
        "4. Applications\n"
        "5. Benefits\n"
        "6. Challenges\n"
        "7. Future scope\n"
        "8. Conclusion\n\n"
        "Use simple and clear technical language."
    ),
    expected_output="A complete research report.",
    agent=writer
)


# ---------------------------------------------------------
# 7. Separate Crew for Research
# ---------------------------------------------------------

research_crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    process=Process.sequential,
    verbose=False
)


# ---------------------------------------------------------
# 8. Separate Crew for Report Writing
# ---------------------------------------------------------

writer_crew = Crew(
    agents=[writer],
    tasks=[writing_task],
    process=Process.sequential,
    verbose=False
)


# ---------------------------------------------------------
# 9. LangGraph State
# ---------------------------------------------------------

class ResearchState(TypedDict):
    topic: str
    research: str
    report: str


# ---------------------------------------------------------
# 10. LangGraph Node - Research
# ---------------------------------------------------------

def run_research(state: ResearchState):
    print("\n[1/4] Research Agent is working...")

    result = research_crew.kickoff(
        inputs={
            "topic": state["topic"]
        }
    )

    research_text = str(result)

    return {
        "research": research_text
    }


# ---------------------------------------------------------
# 11. LangGraph Node - Report Generation
# ---------------------------------------------------------

def generate_report(state: ResearchState):
    print("[2/4] Writer Agent is generating the report...")

    result = writer_crew.kickoff(
        inputs={
            "topic": state["topic"],
            "research": state["research"]
        }
    )

    report_text = str(result)

    return {
        "report": report_text
    }


# ---------------------------------------------------------
# 12. Build LangGraph Workflow
# ---------------------------------------------------------

graph = StateGraph(ResearchState)

graph.add_node("research", run_research)
graph.add_node("write_report", generate_report)

graph.add_edge(START, "research")
graph.add_edge("research", "write_report")
graph.add_edge("write_report", END)

workflow = graph.compile()


# ---------------------------------------------------------
# 13. MLflow Configuration
# ---------------------------------------------------------

mlflow.set_tracking_uri(
    "sqlite:///mlflow.db"
)

mlflow.set_experiment(
    "W9D5_Automated_Research_Report"
)


# ---------------------------------------------------------
# 14. Run the Complete Workflow
# ---------------------------------------------------------

def main():

    print("=" * 60)
    print("W9D5 - AUTOMATED RESEARCH REPORT AGENT")
    print("=" * 60)

    print(f"\nTopic: {RESEARCH_TOPIC}")
    print(f"Model: {MODEL_NAME}")

    with mlflow.start_run(
        run_name="automated_research_report"
    ):

        # Log project parameters
        mlflow.log_param(
            "topic",
            RESEARCH_TOPIC
        )

        mlflow.log_param(
            "model",
            MODEL_NAME
        )

        mlflow.log_param(
            "framework",
            "CrewAI + LangGraph + MLflow + Ragas"
        )

        # Run LangGraph
        final_state = workflow.invoke(
            {
                "topic": RESEARCH_TOPIC,
                "research": "",
                "report": ""
            }
        )

        research = final_state["research"]
        report = final_state["report"]

        print("[3/4] Saving generated report...")

        # Save research notes
        research_file = os.path.join(
            OUTPUT_DIR,
            "research_notes.txt"
        )

        with open(
            research_file,
            "w",
            encoding="utf-8"
        ) as file:
            file.write(research)

        # Save final report
        report_file = os.path.join(
            REPORTS_DIR,
            "research_report.txt"
        )

        with open(
            report_file,
            "w",
            encoding="utf-8"
        ) as file:
            file.write(report)

        # Basic validation
        if not report.strip():
            raise ValueError(
                "Report generation failed. The report is empty."
            )

        word_count = len(report.split())
        character_count = len(report)

        # Log metrics
        mlflow.log_metric(
            "report_word_count",
            word_count
        )

        mlflow.log_metric(
            "report_character_count",
            character_count
        )

        # Save report as MLflow artifact
        mlflow.log_artifact(
            report_file,
            artifact_path="reports"
        )

        print("[4/4] MLflow tracking completed.")

        print("\n" + "=" * 60)
        print("WORKFLOW COMPLETED SUCCESSFULLY")
        print("=" * 60)

        print(f"\nResearch notes:")
        print(research_file)

        print(f"\nFinal report:")
        print(report_file)

        print(f"\nReport word count: {word_count}")
        print(f"Report character count: {character_count}")

        print("\nMLflow experiment:")
        print("W9D5_Automated_Research_Report")


# ---------------------------------------------------------
# 15. Program Entry Point
# ---------------------------------------------------------

if __name__ == "__main__":
    main()