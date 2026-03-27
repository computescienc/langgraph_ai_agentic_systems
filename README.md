# parallel Fully Autonomous Deep Research Agent

**Access at:**
https://github.com/computescienc/langgraph_ai_agentic_systems/blob/main/parallel_workflow/research_agent.py

<img width="1794" height="967" alt="image" src="https://github.com/user-attachments/assets/b58975a8-5c45-47f2-bfcd-6e30ce3b056d" />

<img width="1794" height="967" alt="image" src="https://github.com/user-attachments/assets/5b5f84d4-84f5-41bc-abac-0f99439d5a02" />

<img width="1794" height="967" alt="image" src="https://github.com/user-attachments/assets/3e9aed3c-4ae6-47bc-8eb4-835ca04d656e" />

This system is an advanced autonomous research agent designed to perform comprehensive deep-dives into user-specified topics. Built using the LangGraph framework for workflow orchestration and Streamlit for the user interface, the application automates and personalizes the entire research process from initial inquiry to final document generation.

The architecture follows a sophisticated multi-stage agentic pipeline:

    Planning and Query Decomposition: The system uses a Large Language Model (LLM) to break down a primary research question into multiple specific sub-queries to ensure a broad and deep data collection phase.

    Automated Web Research: Using the Tavily Search API, the agent conducts real-time web searches for each sub-query, gathering current information and source citations from across the live web.

    Multi-Perspective Triangulation: To eliminate cognitive bias, the system processes gathered data through three distinct analytical lenses: an Optimistic perspective (potential and opportunities), a Pessimistic perspective (risks and downsides), and a Balanced perspective (neutral trade-offs).

    Iterative Synthesis and Critique: The agent synthesizes these viewpoints into an initial draft. This draft is then passed to a Critique Node that identifies internal weaknesses, biases, or missing data points before the final version is prepared.

    Human-in-the-Loop Refinement: Unique to this system is the ability for the user to provide direct feedback. Users can request refinements, ask the agent to "continue writing" if a report is cut short, or adjust the narrative focus before finalization.

    Personalized Professional Output: Users can define a custom Analyst Name via the interface, which is dynamically injected into the report’s formal footer and metadata, ensuring the final product is ready for professional distribution.

Technical Features:

    Persistent Research History: Integrated with SQLite, the system automatically saves every research session, allowing users to search, browse, and reload past reports at any time.

    High-Performance Inference: Powered by Groq (Llama 3.3 70B), the backend ensures near-instantaneous processing and streaming of complex analytical tasks.

    Multi-Format Export: Final findings can be exported as raw Markdown or as professionally formatted, auto-paginated PDF documents.

    Real-time Transparency: The UI utilizes Streamlit status containers and text-streaming to show the "thinking process" of the agent at every stage.



**Feel Free to use and update as you need.**
**Access at**: https://github.com/computescienc/langgraph_ai_agentic_systems/blob/main/parallel_workflow/research_agent.py
