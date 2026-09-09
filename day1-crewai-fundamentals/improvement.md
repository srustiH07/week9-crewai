# W9D1 - CrewAI Web Search Improvement

## Objective

The CrewAI research crew was first executed using a local Ollama LLM without web search. The crew was then enhanced by adding a web search tool using `SerperDevTool`.

## Crew Architecture

The crew contains three agents:

1. Researcher
2. Writer
3. Reviewer

The tasks are executed sequentially.

## Version 1 - Without Web Search

File:

`crew_no_web.py`

The Researcher generated research information using the local Ollama model. The Writer converted the research into an article, and the Reviewer evaluated the article.

Output:

`output/crew_output_no_web.txt`

## Version 2 - With Web Search

File:

`crew.py`

The Researcher was enhanced with `SerperDevTool` to retrieve information from the web. The Writer then used the web-based research, and the Reviewer evaluated the resulting article.

Output:

`output/crew_output_with_web.txt`

## Improvement Observed

Adding web search improved the research workflow by allowing the Researcher agent to access current information instead of depending only on the local language model.

The web-enabled version provides:

- Access to current web information
- More relevant research findings
- Better support for real-world examples
- Improved research grounding
- A stronger information base for the Writer

## Review Findings

The Reviewer identified areas that could further improve the generated article:

- Add ethical and legal considerations.
- Include specific real-world case studies.
- Explain additional healthcare applications.
- Distinguish AI from machine learning more clearly.
- Provide more detailed examples and supporting evidence.

## Conclusion

The experiment demonstrates how a basic CrewAI multi-agent workflow can be improved by adding an external web search capability. The Researcher can retrieve current information, the Writer can transform the findings into readable content, and the Reviewer can evaluate the final result.