# W9D2 - CrewAI Tools Web Search Improvement

## Objective

The CrewAI multi-agent research crew was executed in two stages. First, the crew used a local Ollama LLM without external tools. Second, a web search tool was added to the Researcher agent to provide current web-based information.

## Crew Architecture

The crew contains three agents:

1. Researcher
2. Writer
3. Reviewer

The tasks are executed sequentially.

## Version 1 - Without Web Search

File:

`crew_no_web.py`

The Researcher generated information using the local Ollama model. The Writer transformed the research into an article, and the Reviewer evaluated the article.

Output:

`output/crew_output_no_web.txt`

## Version 2 - With Web Search

File:

`crew.py`

The Researcher agent was given the `SerperDevTool` web search tool. It was used to retrieve current information from the web before the Writer generated the article.

Output:

`output/crew_output_with_web.txt`

## Improvement Observed

Adding web search improved the research workflow because the Researcher could access external and current information instead of relying only on the local language model.

The web-enabled version provides:

- Access to current web information
- More detailed research findings
- Better real-world examples
- Stronger research grounding
- More information for the Writer
- A broader basis for the Reviewer

## Output Comparison

The no-web execution produced:

`crew_output_no_web.txt`

The web-enabled execution produced:

`crew_output_with_web.txt`

The web-enabled output contains substantially more research information, demonstrating the effect of adding an external search capability.

## Review Findings

The Reviewer was used to evaluate the generated article and identify areas for improvement.

Possible improvements include:

- Adding specific real-world case studies
- Including ethical and privacy considerations
- Adding supporting evidence for important claims
- Explaining limitations of AI in education
- Discussing bias and fairness
- Improving the distinction between artificial intelligence and machine learning

## Conclusion

The experiment demonstrates how CrewAI tools can extend a multi-agent workflow.

The Researcher uses web search to obtain current information, the Writer converts the findings into readable content, and the Reviewer evaluates the resulting article.

This provides a stronger research workflow than using a local language model alone.