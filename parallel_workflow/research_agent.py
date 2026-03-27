import streamlit as st
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
import time
from fpdf import FPDF
import sqlite3
import json
import datetime
from dotenv import load_dotenv

load_dotenv()


# ====================== DB Setup ======================
def init_db():
    conn = sqlite3.connect("research_history.db")
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS chats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )""")
    conn.commit()
    conn.close()

def save_to_db(query: str, result: dict):
    result_json = json.dumps(result)
    timestamp = datetime.datetime.now().isoformat()
    conn = sqlite3.connect("research_history.db")
    c = conn.cursor()
    c.execute("INSERT INTO chats (query, result_json, timestamp) VALUES (?, ?, ?)",
              (query, result_json, timestamp))
    conn.commit()
    conn.close()

def get_all_chats(search_term: str = ""):
    conn = sqlite3.connect("research_history.db")
    c = conn.cursor()
    if search_term.strip():
        c.execute("""SELECT id, query, timestamp FROM chats 
                     WHERE query LIKE ? ORDER BY timestamp DESC""", (f"%{search_term}%",))
    else:
        c.execute("SELECT id, query, timestamp FROM chats ORDER BY timestamp DESC")
    chats = c.fetchall()
    conn.close()
    return chats

def load_chat(chat_id: int):
    conn = sqlite3.connect("research_history.db")
    c = conn.cursor()
    c.execute("SELECT result_json FROM chats WHERE id = ?", (chat_id,))
    row = c.fetchone()
    conn.close()
    return json.loads(row[0]) if row else None

init_db()

# ====================== State ======================
class ResearchState(TypedDict):
    query: str
    analyst_name: str  # <--- Add this
    sub_queries: List[str]
    search_results: str
    optimistic_analysis: str
    pessimistic_analysis: str
    balanced_analysis: str
    draft_report: str
    critique: str
    final_report: str
    human_feedback: str

# ====================== LLM ======================
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.65, max_tokens=1500)
final_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7, max_tokens=6000)

def stream_llm(prompt: str, placeholder, use_final_llm: bool = False):
    model_to_use = final_llm if use_final_llm else llm
    response = model_to_use.stream([HumanMessage(content=prompt)])
    full_text = ""
    for chunk in response:
        if chunk.content:
            full_text += chunk.content
            placeholder.markdown(full_text + "▌")
            time.sleep(0.012)
    placeholder.markdown(full_text)
    return full_text

# ====================== Nodes ======================
def planner_node(state: ResearchState) -> ResearchState:
    with st.status("🧠 Planning smart sub-questions...", expanded=True):
        prompt = f"""Break down this query into 3-5 specific sub-questions for deep research.\nQuery: {state['query']}\nReturn only a Python list."""
        response = llm.invoke([HumanMessage(content=prompt)])
        try:
            sub_queries = eval(response.content.strip())
            if not isinstance(sub_queries, list):
                sub_queries = [state['query']]
        except:
            sub_queries = [state['query']]
        st.write("Sub-queries:", sub_queries)
        return {"sub_queries": sub_queries}

def multi_search_node(state: ResearchState) -> ResearchState:
    with st.status("🔍 Performing multi-search...", expanded=True):
        all_results = []
        for sq in state.get("sub_queries", [state["query"]])[:5]:
            st.write(f"Searching: {sq}")
            results = TavilySearchResults(max_results=5).invoke(sq)
            all_results.append(f"**Source:** {sq}\n{str(results)}")
        return {"search_results": "\n\n".join(all_results)}

def optimistic_node(state):
    with st.status("🌟 Streaming Optimistic Analysis...", expanded=True):
        ph = st.empty()
        prompt = f"""Optimistic strategist. Deep positive analysis.\nQuery: {state['query']}\nSearch: {state.get('search_results', '')[:4000]}"""
        return {"optimistic_analysis": stream_llm(prompt, ph)}

def pessimistic_node(state):
    with st.status("⚠️ Streaming Risk Analysis...", expanded=True):
        ph = st.empty()
        prompt = f"""Critical risk analyst. Highlight major risks and downsides.\nQuery: {state['query']}\nSearch: {state.get('search_results', '')[:4000]}"""
        return {"pessimistic_analysis": stream_llm(prompt, ph)}

def balanced_node(state):
    with st.status("⚖️ Streaming Balanced Analysis...", expanded=True):
        ph = st.empty()
        prompt = f"""Neutral analyst. Realistic balanced pros/cons.\nQuery: {state['query']}\nSearch: {state.get('search_results', '')[:4000]}"""
        return {"balanced_analysis": stream_llm(prompt, ph)}

def draft_synthesis_node(state):
    with st.status("🔄 Drafting Initial Report...", expanded=True):
        ph = st.empty()
        opt = state.get('optimistic_analysis', '')[:3000]
        pes = state.get('pessimistic_analysis', '')[:3000]
        bal = state.get('balanced_analysis', '')[:3000]
        prompt = f"""Create a detailed draft report synthesizing the three perspectives.

Query: {state['query']}

Optimistic View (summary):
{opt}

Pessimistic View (summary):
{pes}

Balanced View (summary):
{bal}

Write a comprehensive draft with clear sections. Be detailed but stay within reasonable length."""
        return {"draft_report": stream_llm(prompt, ph)}

def critique_node(state):
    with st.status("🧐 Critiquing the draft...", expanded=True):
        ph = st.empty()
        prompt = f"""Critique this draft report. Point out weaknesses, biases, and suggest improvements.\nDraft:\n{state.get('draft_report', '')[:6000]}"""
        return {"critique": stream_llm(prompt, ph)}

# Final Revision Node - Very strict footer control
def final_revision_node(state: ResearchState):
    with st.status("✍️ Producing Complete Final Polished Report...", expanded=True):
        ph = st.empty()
        feedback = state.get('human_feedback', '').strip()
        analyst_name = state.get("analyst_name", "Senior Geopolitical Analyst")

        base_prompt = f"""You are an expert geopolitical analyst. Write a full, complete research report.

**CRITICAL FOOTER INSTRUCTION - FOLLOW EXACTLY:**
You MUST end the report with exactly these two lines and NOTHING after them:

Prepared by: {analyst_name}, Senior Geopolitical Analyst
Date: {datetime.datetime.now().strftime('%d %B %Y')}

Do not repeat the title "Senior Geopolitical Analyst". Use the name exactly as given.

Query: {state['query']}

Information to use:
- Optimistic: {state.get('optimistic_analysis', '')[:2000]}
- Pessimistic: {state.get('pessimistic_analysis', '')[:2000]}
- Balanced: {state.get('balanced_analysis', '')[:2000]}
- Draft: {state.get('draft_report', '')[:4000]}
- Critique: {state.get('critique', '')[:3000]}
- Search: {state.get('search_results', '')[:4000]}

{f"Human Feedback: {feedback}" if feedback else ""}

Rules:
- Structure: Title, Executive Summary, Key Actors, Chronology, Risk Assessment, Scenarios, Conclusion.
- Use markdown and tables where helpful.
- Complete every section. Do not cut off.
- Do NOT add any References section or extra text.
- Stop writing immediately after the footer. No more content after it.

Start directly with the report title."""

        return {"final_report": stream_llm(base_prompt, ph, use_final_llm=True)}

# ====================== Graph ======================
graph = StateGraph(ResearchState)
graph.add_node("planner", planner_node)
graph.add_node("multi_search", multi_search_node)
graph.add_node("optimistic", optimistic_node)
graph.add_node("pessimistic", pessimistic_node)
graph.add_node("balanced", balanced_node)
graph.add_node("draft_synthesis", draft_synthesis_node)
graph.add_node("critique", critique_node)
graph.add_node("final_revision", final_revision_node)

graph.add_edge(START, "planner")
graph.add_edge("planner", "multi_search")
graph.add_edge("planner", "optimistic")
graph.add_edge("planner", "pessimistic")
graph.add_edge("planner", "balanced")
graph.add_edge(["multi_search", "optimistic", "pessimistic", "balanced"], "draft_synthesis")
graph.add_edge("draft_synthesis", "critique")
graph.add_edge("critique", "final_revision")
graph.add_edge("final_revision", END)

workflow = graph.compile()

# ====================== PDF ======================
def create_pdf_report(report_text: str, query: str, analyst_name: str) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=15)
    
    pdf.set_font("Arial", style="B", size=18)
    pdf.cell(0, 12, txt=f"Research Report: {query[:120]}", ln=True, align="C")
    pdf.ln(10)
    
    pdf.set_font("Arial", style="I", size=11)
    pdf.cell(0, 10, txt=f"Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(15)
    
    pdf.set_font("Arial", size=11)
    for line in report_text.splitlines(keepends=True):
        if line.strip():
            pdf.multi_cell(0, 6, txt=line.encode("latin-1", errors="replace").decode("latin-1"))
        else:
            pdf.ln(4)
    
    pdf.set_font("Arial", style="I", size=10)
    pdf.ln(15)
    pdf.cell(0, 8, txt=f"Prepared by: {analyst_name}, Senior Geopolitical Analyst", ln=True, align="C")
    pdf.cell(0, 8, txt=f"Date: {datetime.datetime.now().strftime('%d %B %Y')}", ln=True, align="C")
    
    pdf_bytes = pdf.output(dest="S")
    if isinstance(pdf_bytes, bytearray):
        pdf_bytes = bytes(pdf_bytes)
    return pdf_bytes

# ====================== Streamlit UI ======================
st.set_page_config(page_title="Advanced Research Agent", layout="wide", page_icon="🔬")
st.title("🔬 Fully Autonomous Deep Research Agent")

# Session State
if "current_result" not in st.session_state: st.session_state.current_result = None
if "current_chat_id" not in st.session_state: st.session_state.current_chat_id = None
if "show_history" not in st.session_state: st.session_state.show_history = False
if "history_search" not in st.session_state: st.session_state.history_search = ""
if "show_refinement" not in st.session_state: st.session_state.show_refinement = False
if "analyst_name" not in st.session_state:
    st.session_state.analyst_name = "Haseeb Ahmed"

# Sidebar Analyst Name
st.sidebar.title("👤 Analyst Settings")
analyst_name_input = st.sidebar.text_input("Your Name (for reports)", 
                                           value=st.session_state.analyst_name,
                                           placeholder="Enter your full name")
if analyst_name_input != st.session_state.analyst_name:
    st.session_state.analyst_name = analyst_name_input

# Header Buttons
col1, col2, col3 = st.columns([1, 6, 2])
with col1:
    if st.button("➕ New Research", use_container_width=True):
        st.session_state.current_result = None
        st.session_state.current_chat_id = None
        st.session_state.show_history = False
        st.session_state.show_refinement = False
        st.rerun()

with col3:
    if st.button("📚 History", type="secondary", use_container_width=True):
        st.session_state.show_history = True
        st.session_state.show_refinement = False
        st.rerun()

# History View
if st.session_state.show_history:
    st.subheader("📚 Research History")
    search_term = st.text_input("🔍 Search past researches...", value=st.session_state.history_search)
    st.session_state.history_search = search_term
    chats = get_all_chats(search_term)
    if not chats:
        st.info("No research found yet.")
    else:
        for chat_id, query, ts in chats:
            dt = datetime.datetime.fromisoformat(ts)
            if st.button(f"**{query[:75]}...**\n_{dt.strftime('%b %d, %Y • %H:%M')}_", 
                         key=f"load_{chat_id}", use_container_width=True):
                loaded = load_chat(chat_id)
                if loaded:
                    st.session_state.current_result = loaded
                    st.session_state.current_chat_id = chat_id
                    st.session_state.show_history = False
                    st.rerun()

# Main View
else:
    if st.session_state.current_result is None:
        query = st.text_area("Enter your research question:", height=110,
                             placeholder="What is Imran Khan's latest statement from prison?")
        if st.button("🚀 Run Full Autonomous Research", type="primary", use_container_width=True):
            if query.strip():
                with st.spinner("Running full research pipeline..."):
                    result = workflow.invoke({
                        "query": query,
                        "analyst_name": st.session_state.analyst_name,  # <--- Pass it here
                        "sub_queries": [], 
                        "search_results": "",
                        "optimistic_analysis": "", 
                        "pessimistic_analysis": "",
                        "balanced_analysis": "", 
                        "draft_report": "",
                        "critique": "", 
                        "final_report": "", 
                        "human_feedback": ""
                    })

                    save_to_db(query, result)
                    st.session_state.current_result = result
                    st.rerun()
    else:
        result = st.session_state.current_result
        query = result.get("query", "Research Report")

        st.success(f"✅ Research Complete — {query[:70]}...")

        tabs = st.tabs(["🌟 Optimistic", "⚠️ Risk", "⚖️ Balanced", "🔍 Search Results", "📝 Draft & Critique"])
        with tabs[0]: st.markdown(result.get("optimistic_analysis", ""))
        with tabs[1]: st.markdown(result.get("pessimistic_analysis", ""))
        with tabs[2]: st.markdown(result.get("balanced_analysis", ""))
        with tabs[3]:
            st.subheader("Raw Search Results")
            st.markdown(result.get("search_results", ""))
        with tabs[4]:
            st.subheader("Draft Report")
            st.markdown(result.get("draft_report", ""))
            st.subheader("Critique")
            st.markdown(result.get("critique", ""))

        st.markdown("### 🎛️ Human Review & Final Report")
        col_a, col_b = st.columns(2)

        with col_a:
            if st.button("✅ Generate / Regenerate Complete Final Report", type="primary", use_container_width=True):
                with st.spinner("Generating full complete report..."):
                    # Update the state with the latest name from the sidebar
                    result["analyst_name"] = st.session_state.analyst_name 
                    final_state = final_revision_node(result) # Now only passes result (the state)
                    result["final_report"] = final_state["final_report"]
                    save_to_db(query, result)
                    st.session_state.current_result = result
                    st.rerun()

        with col_b:
            if st.button("🔄 Request Refinement", use_container_width=True):
                st.session_state.show_refinement = True
                st.rerun()

        if st.session_state.get("show_refinement", False):
            st.info("💡 Tell the agent what to improve:")
            human_feedback = st.text_area("Your refinement instructions:", height=130,
                                          placeholder="Continue from where it stopped...")
            col_c, col_d = st.columns(2)
            with col_c:
                if st.button("🚀 Regenerate with Feedback", type="primary"):
                    if human_feedback.strip():
                        with st.spinner("Regenerating report..."):
                            final_state = final_revision_node({**result, "human_feedback": human_feedback}, st.session_state.analyst_name)
                            result["final_report"] = final_state["final_report"]
                            save_to_db(query, result)
                            st.session_state.current_result = result
                            st.session_state.show_refinement = False
                            st.success("✅ Final report updated!")
                            st.rerun()
            with col_d:
                if st.button("Cancel", type="secondary"):
                    st.session_state.show_refinement = False
                    st.rerun()

        if result.get("final_report"):
            st.markdown("### 📊 Final Polished Report")
            st.markdown(result["final_report"])

            if len(result["final_report"]) > 3000:
                if st.button("➕ Continue / Extend Report"):
                    with st.spinner("Extending the report..."):
                        extend_feedback = "Continue writing from where the previous response ended. Complete all remaining sections and add a strong conclusion."
                        final_state = final_revision_node({**result, "human_feedback": extend_feedback}, st.session_state.analyst_name)
                        result["final_report"] += "\n\n" + final_state.get("final_report", "")
                        save_to_db(query, result)
                        st.session_state.current_result = result
                        st.rerun()

            st.subheader("📌 Citations / Sources")
            st.markdown(result.get("search_results", ""))

            col1, col2 = st.columns(2)
            with col1:
                st.download_button("📥 Download as Markdown", 
                                   data=result.get("final_report", ""), 
                                   file_name=f"research_report_{int(time.time())}.md", 
                                   mime="text/markdown", use_container_width=True)
            with col2:
                pdf_data = create_pdf_report(result.get("final_report", ""), query, st.session_state.analyst_name)
                st.download_button("📄 Download as Pure PDF", 
                                   data=pdf_data, 
                                   file_name=f"research_report_{int(time.time())}.pdf", 
                                   mime="application/pdf", use_container_width=True)

st.caption("Advanced Research Agent • Token-Safe • Custom Analyst Name")
