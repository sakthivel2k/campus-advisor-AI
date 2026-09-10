"""
CampusMate AI - LangGraph Agent Workflow Module
Constructs the LangGraph StateGraph with nodes for the Agent, Tools (including RAG),
and Checkpointing for persistent conversation memory.
"""

import os
import sys
import warnings
import logging
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv

# Suppress provider and library warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
os.environ["ABSL_LOGGING_MIN_LEVEL"] = "error"

warnings.filterwarnings("ignore")
warnings.showwarning = lambda *args, **kwargs: None
logging.captureWarnings(True)

for logger_name in [
    "langchain",
    "langchain_core",
    "langchain_community",
    "langchain_google_genai",
    "google",
    "grpc",
]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from tools import ALL_TOOLS

# Load environment variables
load_dotenv()


# =========================================================
# 1. State Definition
# =========================================================
class AgentState(TypedDict):
    """LangGraph conversation state carrying the message history."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


# =========================================================
# 2. LLM Initialization with Multi-Provider Support
# =========================================================
def get_llm():
    """
    Initialize a LangChain-compatible chat model based on environment configuration.
    Supports OpenAI, Groq, and Google Gemini with automatic fallback detection.
    """
    provider = os.getenv("LLM_PROVIDER", "").lower().strip()
    model_name = os.getenv("MODEL_NAME", "").strip()

    # Provider: Groq
    if provider == "groq" or (not provider and os.getenv("GROQ_API_KEY")):
        try:
            from langchain_groq import ChatGroq
            chosen_model = model_name or "llama-3.3-70b-versatile"
            return ChatGroq(
                model_name=chosen_model,
                temperature=0.2,
                api_key=os.getenv("GROQ_API_KEY")
            )
        except Exception as e:
            if provider == "groq":
                raise e

    # Provider: Google Gemini
    if provider == "google" or (not provider and os.getenv("GOOGLE_API_KEY")):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            chosen_model = model_name or "gemini-1.5-flash"
            return ChatGoogleGenerativeAI(
                model=chosen_model,
                temperature=0.2,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
        except Exception as e:
            if provider == "google":
                raise e

    # Provider: OpenAI (Default)
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key or provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
            chosen_model = model_name or "gpt-4o-mini"
            return ChatOpenAI(
                model=chosen_model,
                temperature=0.2,
                api_key=openai_key
            )
        except Exception as e:
            raise e

    # No key found
    raise ValueError(
        "No LLM API Key detected. Please set OPENAI_API_KEY, GROQ_API_KEY, "
        "or GOOGLE_API_KEY in your .env file."
    )


# System Prompt configuring CampusMate AI persona and tool instructions
SYSTEM_PROMPT = SystemMessage(
    content="""You are CampusMate AI, an intelligent, empathetic, and highly accurate AI Student Support Assistant designed for college students.

Your Capabilities & Guidelines:
1. College Information & RAG Retrieval:
   - Whenever the student asks about college regulations, attendance rules, syllabus, subjects, library hours, canteen, fees, exams, or campus FAQs, ALWAYS use the 'search_college_knowledge_base' tool to retrieve verified facts from the college documents.
   - Do NOT fabricate college policies; rely on retrieved document evidence.
   - Always cite the key details (e.g., minimum 75% attendance requirement, specific semester courses, library hours).

2. Date and Time:
   - If the student asks for the current date, time, or day, use the 'get_current_datetime' tool.

3. Attendance Calculation:
   - If the student provides attended classes and total classes, use the 'calculate_attendance_status' tool.

4. Conversation Memory & Context:
   - Retain and use details shared earlier in the conversation (e.g., the student's department, semester, student name, or specific subject interests).
   - If a student previously mentioned "My department is Computer Science" and later asks "What are my subjects?", answer for Computer Science Semester 5/6.

5. Tone:
   - Maintain a friendly, supportive, and professional tone.
   - Format lists with clean bullet points.
"""
)


# =========================================================
# 3. LangGraph Workflow Definition
# =========================================================
def create_campusmate_graph():
    """
    Builds the LangGraph state machine:
    
    START -> agent_node -> (tools_condition)
                               ├── tool_calls present -> tools_node -> agent_node
                               └── no tool_calls     -> END
    """
    llm = get_llm()
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    def agent_node(state: AgentState):
        """Processes conversation messages, prepends system prompt, and invokes LLM."""
        messages = [SYSTEM_PROMPT] + list(state["messages"])
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # Initialize LangGraph StateGraph
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(ALL_TOOLS))

    # Add Edges
    workflow.add_edge(START, "agent")

    # Conditional routing: If LLM generated tool calls, go to 'tools', else finish at END
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            END: END
        }
    )

    # After tools execute, return control back to the agent to synthesize final response
    workflow.add_edge("tools", "agent")

    # Checkpointer for conversation state memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app
