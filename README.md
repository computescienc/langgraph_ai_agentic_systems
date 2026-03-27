# parallel Fully Autonomous Deep Research Agent


<img width="1794" height="967" alt="image" src="https://github.com/user-attachments/assets/b58975a8-5c45-47f2-bfcd-6e30ce3b056d" />

<img width="1794" height="967" alt="image" src="https://github.com/user-attachments/assets/5b5f84d4-84f5-41bc-abac-0f99439d5a02" />

<img width="1794" height="967" alt="image" src="https://github.com/user-attachments/assets/3e9aed3c-4ae6-47bc-8eb4-835ca04d656e" />

This system is an advanced autonomous research agent designed to perform comprehensive deep-dives into user-specified topics. Built using the LangGraph framework for workflow orchestration and Streamlit for the user interface, the application automates the entire research process from initial inquiry to final document generation.

The architecture follows a multi-stage agentic pipeline:

Planning and Query Decomposition: The system uses a Large Language Model (LLM) to break down a primary research question into multiple specific sub-queries to ensure a broad data collection phase.

Automated Web Research: Using the Tavily Search API, the agent conducts real-time web searches for each sub-query, gathering current information and source citations.

Multi-Perspective Analysis: To avoid bias, the system processes the gathered data through three distinct analytical lenses: an optimistic perspective (focusing on benefits and potential), a pessimistic perspective (focusing on risks and downsides), and a balanced perspective (focusing on neutral, evidence-based trade-offs).

Iterative Synthesis and Critique: The agent synthesizes these viewpoints into an initial draft. This draft is then passed to a critique node that identifies weaknesses, biases, or missing information.

Final Report Generation: The system incorporates the critique and original search data to produce a polished final report, complete with citations.

Technical features include real-time response streaming for transparency during the thinking process, a session-based research history sidebar, and the ability to export the final findings as either Markdown or professionally formatted PDF files. The backend is powered by high-performance models via the Groq inference engine, ensuring rapid processing of complex analytical tasks.
