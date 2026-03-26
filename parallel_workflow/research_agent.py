import streamlit as st
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv
import time
from fpdf import FPDF
load_dotenv()

# ====================== State ======================
class ResearchState(TypedDict):
    query: str
    sub_queries: List[str]
    search_results: str
    optimistic_analysis: str
    pessimistic_analysis: str
    balanced_analysis: str
    draft_report: str
    critique: str
    final_report: str

# ====================== LLM ======================
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.65,
    max_tokens=1500
)

search_tool = TavilySearchResults(max_results=5)

# ====================== Streaming Helper ======================
def stream_llm(prompt: str, placeholder):
    response = llm.stream([HumanMessage(content=prompt)])
    full_text = ""
    for chunk in response:
        if chunk.content:
            full_text += chunk.content
            placeholder.markdown(full_text + "▌")
            time.sleep(0.02)
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
            results = search_tool.invoke(sq)
            all_results.append(f"**Source:** {sq}\n{str(results)}")
        return {"search_results": "\n\n".join(all_results)}

def optimistic_node(state): 
    with st.status("🌟 Streaming Optimistic Analysis...", expanded=True):
        ph = st.empty()
        prompt = f"""Optimistic strategist. Deep positive analysis.\nQuery: {state['query']}\nSearch: {state.get('search_results', '')}"""
        return {"optimistic_analysis": stream_llm(prompt, ph)}

def pessimistic_node(state): 
    with st.status("⚠️ Streaming Risk Analysis...", expanded=True):
        ph = st.empty()
        prompt = f"""Critical risk analyst. Highlight major risks and downsides.\nQuery: {state['query']}\nSearch: {state.get('search_results', '')}"""
        return {"pessimistic_analysis": stream_llm(prompt, ph)}

def balanced_node(state): 
    with st.status("⚖️ Streaming Balanced Analysis...", expanded=True):
        ph = st.empty()
        prompt = f"""Neutral analyst. Realistic balanced pros/cons.\nQuery: {state['query']}\nSearch: {state.get('search_results', '')}"""
        return {"balanced_analysis": stream_llm(prompt, ph)}

def draft_synthesis_node(state): 
    with st.status("🔄 Drafting Initial Report...", expanded=True):
        ph = st.empty()
        prompt = f"""Create a detailed draft report using the three perspectives.\nQuery: {state['query']}\nOptimistic: {state.get('optimistic_analysis', '')}\nPessimistic: {state.get('pessimistic_analysis', '')}\nBalanced: {state.get('balanced_analysis', '')}"""
        return {"draft_report": stream_llm(prompt, ph)}

def critique_node(state): 
    with st.status("🧐 Critiquing the draft...", expanded=True):
        ph = st.empty()
        prompt = f"""Critique this draft report. Point out weaknesses, biases, and suggest improvements.\nDraft:\n{state.get('draft_report', '')}"""
        return {"critique": stream_llm(prompt, ph)}

def final_revision_node(state): 
    with st.status("✍️ Producing Final Report...", expanded=True):
        ph = st.empty()
        prompt = f"""Improve the draft using the critique and produce the final polished report. Include citations from search results where possible.\nQuery: {state['query']}\nDraft: {state.get('draft_report', '')}\nCritique: {state.get('critique', '')}\nSearch Results: {state.get('search_results', '')[:3000]}"""
        return {"final_report": stream_llm(prompt, ph)}

def create_pdf_report(report_text: str, query: str) -> bytes:
    """Generate a clean, professional PDF - Fixed bytearray issue"""
    pdf = FPDF()
    pdf.add_page()
    
    # Set nice margins
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=15)

    # ==================== TITLE ====================
    pdf.set_font("Arial", style="B", size=18)
    safe_title = query[:120]
    pdf.cell(0, 12, txt=f"Research Report: {safe_title}", ln=True, align="C")
    pdf.ln(10)

    # Timestamp
    pdf.set_font("Arial", style="I", size=11)
    pdf.cell(0, 10, txt=f"Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(15)

    # ==================== REPORT BODY ====================
    pdf.set_font("Arial", size=11)
    
    def safe_text(text: str) -> str:
        return text.encode("latin-1", errors="replace").decode("latin-1")

    for line in report_text.splitlines(keepends=True):
        if line.strip():
            pdf.multi_cell(0, 6, txt=safe_text(line))
        else:
            pdf.ln(4)

    # ==================== CRITICAL FIX ====================
    pdf_bytes = pdf.output(dest="S")
    # Convert bytearray → bytes (this fixes the Streamlit error)
    if isinstance(pdf_bytes, bytearray):
        pdf_bytes = bytes(pdf_bytes)

    return pdf_bytes

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

# ====================== Streamlit UI ======================
st.set_page_config(page_title="Advanced Research Agent", layout="wide", page_icon="🔬")
st.title("🔬 Fully Autonomous Deep Research Agent")

# Sidebar History
st.sidebar.title("📚 Research History")
if "research_history" not in st.session_state:
    st.session_state.research_history = []

for i, item in enumerate(st.session_state.research_history[-10:]):
    if st.sidebar.button(f"{item['query'][:55]}...", key=f"hist_{i}"):
        st.session_state.last_result = item["result"]
        st.session_state.show_approval = True

query = st.text_area("Enter your research question:", height=110,
                     placeholder="What is Imran Khan's latest statement from prison?")

if st.button("🚀 Run Full Autonomous Research", type="primary", use_container_width=True):
    if query.strip():
        with st.spinner("Running full research pipeline..."):
            result = workflow.invoke({
                "query": query,
                "sub_queries": [], "search_results": "",
                "optimistic_analysis": "", "pessimistic_analysis": "", "balanced_analysis": "",
                "draft_report": "", "critique": "", "final_report": ""
            })
        st.session_state.last_result = result
        st.session_state.show_approval = True
        st.rerun()

# ====================== DISPLAY LOGIC (Outside button) ======================
if st.session_state.get("show_approval", False) and "last_result" in st.session_state:
    result = st.session_state.last_result

    st.success("✅ All perspectives generated! Review them below.")

    tab1, tab2, tab3, tab4 = st.tabs(["🌟 Optimistic", "⚠️ Risk", "⚖️ Balanced", "🔍 Search Results"])
    with tab1: st.markdown(result.get("optimistic_analysis", ""))
    with tab2: st.markdown(result.get("pessimistic_analysis", ""))
    with tab3: st.markdown(result.get("balanced_analysis", ""))
    with tab4:
        st.subheader("Raw Search Results")
        st.markdown(result.get("search_results", "No search data"))

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Approve & Generate Final Report", type="primary"):
            st.session_state.show_final = True
            st.rerun()
    with col2:
        if st.button("🔄 Revise Again (Extra Critique)"):
            st.info("Extra revision coming in next update.")

# ====================== FINAL REPORT ======================
if st.session_state.get("show_final", False) and "last_result" in st.session_state:
    result = st.session_state.last_result

    st.markdown("### 📊 Final Polished Report")
    st.markdown(result.get("final_report", "No report generated"))

    # Citations
    st.subheader("📌 Citations / Sources")
    st.markdown(result.get("search_results", "No citations available"))

    # ====================== DOWNLOAD BUTTONS ======================
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📥 Download as Markdown",
            data=result.get("final_report", ""),
            file_name=f"research_report_{int(time.time())}.md",
            mime="text/markdown",
            use_container_width=True
        )

    with col2:
        # Generate pure PDF
        pdf_data = create_pdf_report(
            result.get("final_report", "No report generated"),
            result.get("query", "Research Report")
        )
        st.download_button(
            label="📄 Download as Pure PDF",
            data=pdf_data,
            file_name=f"research_report_{int(time.time())}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    # Save to history
    st.session_state.research_history.append({
        "query": result["query"],
        "result": result
    })

st.caption("Advanced Research Agent with Citations & PDF Support")