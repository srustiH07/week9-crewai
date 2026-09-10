# W9D3 - Multi-Agent Research Crew

## Objective

The objective of this task is to build a multi-agent research pipeline using CrewAI.

The pipeline contains three agents:

1. Researcher
2. Writer
3. Reviewer

The agents work sequentially:

Researcher → Writer → Reviewer

---

## Run 1 - Without Web Search

The first version uses a local Ollama `qwen2.5:3b` model without an external web search tool.

The Researcher performs research using the model's available knowledge. The Writer converts the research into an article, and the Reviewer checks the final article.

### Observations

- The multi-agent workflow worked successfully.
- The Researcher generated useful information from available model knowledge.
- The Writer converted the research into a structured article.
- The Reviewer identified missing details and suggested improvements.
- The output may not contain the latest web information.
- Real-world examples and recent developments may be limited.

---

## Run 2 - With Web Search

The second version adds `SerperDevTool` to the Researcher agent.

The Researcher can now search the web for current information before passing the findings to the Writer.

The Writer creates the article using the research results, and the Reviewer evaluates the generated article.

### Observations

- The web-search pipeline executed successfully.
- The Researcher obtained information using web search.
- The article contained more detailed applications, challenges and examples.
- The output was more informative and useful for a research task.
- Web search improves access to current information.

---

## Comparison

### Without Web Search

- Uses local Ollama model knowledge.
- Does not access current web information.
- Suitable for basic research and offline demonstrations.
- May have limited recent examples.

### With Web Search

- Uses SerperDevTool for web search.
- Provides access to current web information.
- Produces more detailed research.
- Provides additional examples and supporting information.
- More suitable for research tasks requiring recent information.

---

## Multi-Agent Pipeline

The W9D3 pipeline follows a sequential multi-agent architecture:

Researcher
↓
Writer
↓
Reviewer

### Researcher

Collects and organizes information about the given topic.

### Writer

Uses the Researcher's findings to create a clear and structured article.

### Reviewer

Checks the article for accuracy, clarity, completeness and usefulness and provides improvement suggestions.

---

## Overall Improvement

Adding web search improved the research pipeline because the Researcher could access current information instead of relying only on the local language model's existing knowledge.

The combination of web search and multiple specialized agents makes the workflow more useful for research-oriented applications.

The Reviewer also helps identify weaknesses and areas that can be improved in the final output.

---

## Conclusion

The W9D3 task successfully demonstrates a CrewAI multi-agent research pipeline.

The comparison between the two executions shows that adding web search can improve the depth, relevance and usefulness of research output.

The Researcher → Writer → Reviewer architecture also demonstrates how different agents can collaborate to complete a larger task sequentially.