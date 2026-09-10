# W9D2 - CrewAI Tools

## Objective

Implement a CrewAI multi-agent research workflow using web search and evaluate the improvement in output quality.

## Agents

- Researcher - performs research and gathers information
- Writer - converts research into an article
- Reviewer - evaluates the generated article

## Tools and Technologies

- CrewAI
- CrewAI Tools
- SerperDevTool
- Ollama
- Qwen2.5:3b
- Python

## Execution Versions

### Without Web Search

`crew_no_web.py`

Output:

`output/crew_output_no_web.txt`

### With Web Search

`crew.py`

Output:

`output/crew_output_with_web.txt`

## Improvement

The web-enabled version allows the Researcher to obtain external and current information, providing a stronger information base for the Writer and Reviewer.

## Files

- `crew_no_web.py` - CrewAI workflow without web search
- `crew.py` - CrewAI workflow with web search
- `improvement.md` - comparison and improvement documentation
- `output/` - execution results