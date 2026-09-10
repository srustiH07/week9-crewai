# W9D5 - Ragas Evaluation
# Lightweight local evaluation using Ragas + Ollama.

import os

import mlflow
from datasets import Dataset
from langchain_ollama import ChatOllama

from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness
from ragas.run_config import RunConfig


# ---------------------------------------------------------
# 1. Paths
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

REPORT_FILE = os.path.join(
    BASE_DIR,
    "reports",
    "research_report.txt"
)

RESEARCH_FILE = os.path.join(
    BASE_DIR,
    "output",
    "research_notes.txt"
)

EVALUATION_FILE = os.path.join(
    BASE_DIR,
    "output",
    "ragas_evaluation.txt"
)

MLFLOW_DB = os.path.join(
    BASE_DIR,
    "mlflow.db"
)


# ---------------------------------------------------------
# 2. Check generated files
# ---------------------------------------------------------

if not os.path.exists(REPORT_FILE):
    raise FileNotFoundError(
        "Research report not found. Run research_report_agent.py first."
    )

if not os.path.exists(RESEARCH_FILE):
    raise FileNotFoundError(
        "Research notes not found. Run research_report_agent.py first."
    )


# ---------------------------------------------------------
# 3. Read report and research
# ---------------------------------------------------------

with open(REPORT_FILE, "r", encoding="utf-8") as file:
    report = file.read()

with open(RESEARCH_FILE, "r", encoding="utf-8") as file:
    research = file.read()


if not report.strip():
    raise ValueError("Research report is empty.")

if not research.strip():
    raise ValueError("Research notes are empty.")


# ---------------------------------------------------------
# 4. Use a smaller evaluation sample
# ---------------------------------------------------------

# A smaller portion reduces the amount of text sent to
# the local Ollama evaluator.

evaluation_context = research[:3000]
evaluation_response = report[:3000]


# ---------------------------------------------------------
# 5. Configure local Ollama evaluator
# ---------------------------------------------------------

ollama_llm = ChatOllama(
    model="qwen2.5:3b",
    base_url="http://localhost:11434",
    temperature=0.0
)

evaluator_llm = LangchainLLMWrapper(
    ollama_llm
)


# ---------------------------------------------------------
# 6. Create Ragas dataset
# ---------------------------------------------------------

dataset = Dataset.from_dict(
    {
        "user_input": [
            "Explain Artificial Intelligence in Healthcare."
        ],

        "response": [
            evaluation_response
        ],

        "retrieved_contexts": [
            [evaluation_context]
        ]
    }
)


# ---------------------------------------------------------
# 7. Select one lightweight Ragas metric
# ---------------------------------------------------------

metrics = [
    Faithfulness(
        llm=evaluator_llm
    )
]


# ---------------------------------------------------------
# 8. Run Ragas
# ---------------------------------------------------------

print("=" * 60)
print("RAGAS EVALUATION")
print("=" * 60)

print("\nEvaluating report faithfulness...")
print("Using local Ollama model: qwen2.5:3b")


run_config = RunConfig(
    timeout=120,
    max_retries=0,
    max_workers=1
)


results = evaluate(
    dataset=dataset,
    metrics=metrics,
    run_config=run_config
)


# ---------------------------------------------------------
# 9. Display results
# ---------------------------------------------------------

results_dataframe = results.to_pandas()

print("\nEvaluation results:")
print(results_dataframe)


# ---------------------------------------------------------
# 10. Save results
# ---------------------------------------------------------

with open(
    EVALUATION_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write("W9D5 RAGAS EVALUATION\n")
    file.write("=" * 50 + "\n\n")

    file.write(
        results_dataframe.to_string(
            index=False
        )
    )

    file.write("\n\n")
    file.write("Metric used:\n")
    file.write("- Faithfulness\n")


# ---------------------------------------------------------
# 11. Configure MLflow using SQLite
# ---------------------------------------------------------

mlflow.set_tracking_uri(
    "sqlite:///" + MLFLOW_DB.replace("\\", "/")
)

mlflow.set_experiment(
    "W9D5_Automated_Research_Report"
)


# ---------------------------------------------------------
# 12. Log Ragas result to MLflow
# ---------------------------------------------------------

with mlflow.start_run(
    run_name="ragas_evaluation"
):

    mlflow.log_param(
        "evaluation_model",
        "qwen2.5:3b"
    )

    mlflow.log_param(
        "evaluation_framework",
        "Ragas"
    )

    mlflow.log_param(
        "metric",
        "Faithfulness"
    )

    for column in results_dataframe.columns:

        value = results_dataframe[column].iloc[0]

        if isinstance(value, (int, float)):

            if value == value:

                mlflow.log_metric(
                    column,
                    float(value)
                )

    mlflow.log_artifact(
        EVALUATION_FILE,
        artifact_path="evaluation"
    )


# ---------------------------------------------------------
# 13. Final message
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("RAGAS EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nEvaluation file:")
print(EVALUATION_FILE)

print("\nMLflow tracking completed.")