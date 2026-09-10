# 🎓 CAMPUSMATE AI — AI-Powered Student Support Assistant

> **IBM Naan Mudhalvan Internship Project**  
> *A Production-Ready AI Agent implementing LangChain, LangGraph, RAG, Tool Calling, and Conversation State Memory in an Interactive Terminal Interface.*

---

## 📌 Project Overview

**CampusMate AI** is an intelligent, conversational academic assistant engineered for college students. It provides instant, accurate, and context-aware responses to queries regarding **college regulations, attendance policies, course syllabi, library hours, examination rules, and campus FAQs**.

Rather than relying on static or hallucinated responses, CampusMate AI leverages an intelligent **LangGraph Agentic Workflow** coupled with **Retrieval-Augmented Generation (RAG)** over official college documentation and custom **LangChain Tools**.

---

## 🎯 Problem Statement

In collegiate environments:
1. Students frequently struggle to locate authoritative information spread across lengthy handbooks, circulars, and syllabi.
2. Academic administrative offices spend excessive hours answering repetitive inquiries regarding attendance cutoffs, syllabus outlines, and examination deadlines.
3. Traditional keyword search systems fail to understand context, multi-turn follow-up questions, or student-specific situations (e.g., remembering a student's department across questions).

---

## 💡 Solution

**CampusMate AI** resolves this by providing a unified AI Agent that:
- **Understands Multi-Turn Dialogue**: Maintains conversation state and context using **LangGraph Memory Checkpointing**.
- **Delivers Grounded Answers**: Searches indexed college documents using **FAISS Vector Database & RAG**.
- **Executes Dynamic Tools**: Interactively checks real-time dates/times, computes attendance percentages, and evaluates exam eligibility.
- **Operates in a Clean Terminal Interface**: Requires zero complex web setup — runs seamlessly on any terminal.

---

## 🚀 Key Features

| Feature | Description |
| :--- | :--- |
| **🧠 LangGraph Workflow** | Cyclic state-machine agent orchestrating reasoning, tool calling, and RAG retrieval. |
| **📚 Document RAG** | Document loading, chunking, and similarity search powered by FAISS and HuggingFace/OpenAI embeddings. |
| **🛠️ Tool Calling** | Dynamic invocation of system date/time tools, RAG search, and attendance calculator. |
| **💾 Conversation Memory** | LangGraph `MemorySaver` preserves context (e.g., student department, name) across questions. |
| **⚡ Multi-LLM Support** | Compatible with **OpenAI (GPT-4o-mini)**, **Groq (Llama 3.3)**, and **Google Gemini**. |
| **🛡️ Robust Error Handling** | Beginner-friendly messages for missing keys, input validation, and vector store auto-generation. |

---

## 🏗️ Project Architecture

```
Student (Terminal User)
         │
         ▼
 ┌─────────────────────────────────────────────────────────────┐
 │                    Terminal CLI Interface                   │
 │                     (Interactive Loop)                      │
 └──────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │                  LangGraph Workflow (StateGraph)            │
 │                                                             │
 │   START ──► [ Agent Node (LLM Reasoning) ]                  │
 │                     │                                       │
 │             (Tool Call Required?)                           │
 │             ├── YES ──► [ Tools Node ]                      │
 │             │                 ├── RAG Vector Search         │
 │             │                 ├── System Date / Time        │
 │             │                 └── Attendance Calculator     │
 │             │                 └── (Returns to Agent Node)   │
 │             │                                               │
 │             └── NO  ──► [ END (Final Response) ]            │
 │                                                             │
 │   State & Checkpoint Memory: MemorySaver (Thread Context)   │
 └──────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │                      RAG Vector Engine                      │
 │      FAISS Index ◄── Recursive Splitter ◄── data/*.txt      │
 └─────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
campusmate-ai/
│
├── main.py               # Interactive terminal interface & application entrypoint
├── agent.py              # LangGraph StateGraph, agent node, and checkpointer setup
├── rag.py                # Document loader, chunker, embeddings, and FAISS vector store
├── tools.py              # LangChain @tool definitions (RAG search, DateTime, Attendance)
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variables template
├── README.md             # Comprehensive project documentation
│
└── data/                 # College knowledge base documents (Demo data)
    ├── college_regulations.txt   # Attendance rules, exam guidelines, policies
    ├── faq.txt                   # Library hours, portal access, dining, bonafide
    └── syllabus.txt              # CSE, IT, ECE course curricula & credits
```

---

## 🔬 How Core Concepts Are Implemented

### 1. LangChain
- **Tool Creation**: LangChain's `@tool` decorator wraps custom functions into structured schemas with docstrings that the LLM understands.
- **Document Processing**: `RecursiveCharacterTextSplitter` chunks documents intelligently preserving semantic boundaries.
- **Model Bindings**: `llm.bind_tools(ALL_TOOLS)` binds schema definitions directly to the model.

### 2. LangGraph
- **State Definition**: Uses `AgentState` with `Annotated[Sequence[BaseMessage], add_messages]` to automatically accumulate chat history.
- **Nodes & Edges**:
  - `agent`: Prepends system guidelines and prompts the LLM.
  - `tools`: Prebuilt `ToolNode` executes tool calls.
  - `tools_condition`: Inspects LLM output and routes to `tools` if tool invocations are requested, or transitions to `END`.
- **Checkpointing**: `MemorySaver` tracks session state keyed by `thread_id`, providing seamless multi-turn memory without manual message buffer management.

### 3. Retrieval-Augmented Generation (RAG)
1. **Source Loading**: Loads `.txt` files from `data/` (auto-created if missing).
2. **Chunking**: Chunks text with 500 characters and 100 character overlap.
3. **Embeddings & Vector Store**: Stores embeddings in a local **FAISS** vector database.
4. **Retrieval**: Queries vector store via similarity search (`k=3`) and feeds verified context back to the AI Agent.

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.10 to 3.13 installed on your system.

### 2. Clone / Navigate to the Project
```bash
cd campusmate-ai
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys
Create your `.env` file from the example:
```bash
# On Windows (PowerShell):
Copy-Item .env.example .env

# On Linux/macOS:
cp .env.example .env
```

Open `.env` and paste your preferred LLM API key:
```ini
# Choose at least ONE:
OPENAI_API_KEY=sk-proj-xxxxxx

# OR use a free Groq key (https://console.groq.com):
# GROQ_API_KEY=gsk_xxxxxx

# OR use a free Google Gemini key (https://aistudio.google.com):
# GOOGLE_API_KEY=AIzaxxxxxx
```

---

## ▶️ How to Run

Launch the interactive chatbot in your terminal:

```bash
python main.py
```

### Available Terminal Commands:
- `help` — View available commands and example prompts.
- `clear` — Clear the terminal screen.
- `status` — Check active session thread ID and memory state.
- `exit` or `quit` — Exit the assistant.

---

## 💬 Sample Terminal Conversation

```text
=================================================
                CAMPUSMATE AI
       Intelligent Student Support Assistant
=================================================

Hello! I can help you with college-related questions.

Ask me about:

• Attendance
• Regulations
• Syllabus
• College FAQs
• Academic information

Type 'help' for commands.
Type 'exit' to quit.

You: What is the minimum attendance required to write semester exams?

CampusMate AI is thinking...
  [Tool Invoked: search_college_knowledge_base]

Assistant:
According to the official College Academic Regulations:
• The minimum mandatory attendance required across all registered courses is 75%.
• If your attendance is between 65% and 74% due to verified medical emergencies or official duty, you may apply for condonation subject to approval.
• Students with less than 75% attendance without approved condonation are NOT eligible to appear for end-semester examinations and must redo the course.

-------------------------------------------------------

You: My department is Computer Science.

Assistant:
Got it! I have noted that your department is Computer Science and Engineering (CSE). How can I assist you with your curriculum, regulations, or campus queries?

-------------------------------------------------------

You: What subjects are included in Semester 5?

CampusMate AI is thinking...
  [Tool Invoked: search_college_knowledge_base]

Assistant:
For Computer Science and Engineering (CSE) Semester 5, the curriculum includes:
1. CS501: Artificial Intelligence (4 Credits)
2. CS502: Database Management Systems (4 Credits)
3. CS503: Computer Networks (3 Credits)
4. CS504: Software Engineering (3 Credits)
5. CS505: Machine Learning Fundamentals (4 Credits)
6. CS506: Database Management Systems Lab (2 Credits)
7. CS507: Artificial Intelligence & Machine Learning Lab (2 Credits)

-------------------------------------------------------

You: I attended 28 out of 40 classes. Am I eligible for exams?

CampusMate AI is thinking...
  [Tool Invoked: calculate_attendance_status]

Assistant:
Here is your attendance evaluation:
• Classes Attended: 28 / 40
• Attendance Percentage: 70.00%
• Status: SHORTAGE / NOT ELIGIBLE (Minimum required is 75.00%)

⚠ Warning: You currently have an attendance shortage. You need to attend the next 8 consecutive classes without missing any to reach the 75% eligibility mark. You may also consult your HOD regarding medical condonation if applicable.

-------------------------------------------------------

You: exit

Thank you for using CampusMate AI. Have a great academic semester!
```

---

## 🏆 Assessment Criteria Alignment (IBM Naan Mudhalvan)

- ✅ **LangChain Integration**: Built using `langchain-core` tools, document loaders, and text splitters.
- ✅ **LangGraph Implementation**: Real cyclic `StateGraph` with agent nodes, conditional edge routing (`tools_condition`), and `MemorySaver`.
- ✅ **RAG Pipeline**: Production-style chunking, embedding generation, and local FAISS vector search.
- ✅ **Agent Reasoning & Tool Calling**: Autonomous decision-making on when to search knowledge base vs. run system tools.
- ✅ **Conversation Memory**: Remembers user context (department, past inquiries) across conversation turns.
- ✅ **Zero Over-Engineering**: Clean, modular, standalone terminal app that runs out-of-the-box.
