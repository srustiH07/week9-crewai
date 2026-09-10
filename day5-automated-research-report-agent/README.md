# W9D5 - Automated Research Report Agent

## Objective

Build an automated research report workflow using the approved AI/ML stack:

- CrewAI
- LangGraph
- MLflow
- Ragas
- MLOps practices

## Architecture

User Topic
    |
    v
LangGraph Workflow
    |
    v
CrewAI Research Agent
    |
    v
CrewAI Writer Agent
    |
    v
Generated Research Report
    |
    v
Ragas Evaluation
    |
    v
MLflow Tracking

## Components

### CrewAI

CrewAI provides two specialized agents:

1. Research Analyst
2. Report Writer

The research agent creates structured research notes, while the writer converts those notes into a complete report.

### LangGraph

LangGraph controls the workflow:

Research Agent -> Writer Agent -> Final Report

### Ragas

Ragas is used to evaluate the generated report using the Faithfulness metric.

The evaluation uses the local Ollama model:

qwen2.5:3b

### MLflow

MLflow tracks:

- Project parameters
- Report word count
- Report character count
- Ragas evaluation results
- Generated report artifacts

A local SQLite database is used for MLflow tracking.

## Output Files

```text
output/
    research_notes.txt
    ragas_evaluation.txt

reports/
    research_report.txt