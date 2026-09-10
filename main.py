import os
import sys
import uuid
import warnings
import logging

# ----------------------------------------------------------------------
# Suppress unnecessary library warnings and background logs
# ----------------------------------------------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["DATASETS_VERBOSITY"] = "error"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
os.environ["ABSL_LOGGING_MIN_LEVEL"] = "error"

warnings.filterwarnings("ignore")
warnings.showwarning = lambda *args, **kwargs: None
logging.captureWarnings(True)

for logger_name in [
    "transformers",
    "sentence_transformers",
    "huggingface_hub",
    "langchain",
    "langchain_core",
    "langchain_community",
    "langchain_google_genai",
    "google",
    "grpc",
    "httpx",
    "httpcore",
    "urllib3",
]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# Enable ANSI escape codes on Windows console
if sys.platform == "win32":
    os.system("")

BANNER = """
=================================================
                CAMPUSMATE AI
       Intelligent Student Support Assistant
=================================================

Hello! I can help you with college-related questions.

Ask me about:

* Attendance
* Regulations
* Syllabus
* College FAQs
* Academic information

Type 'help' for commands.
Type 'exit' to quit.
"""

HELP_TEXT = """
-------------------------------------------------
CAMPUSMATE AI - COMMANDS & SAMPLE QUESTIONS
-------------------------------------------------

Special Commands:
  help    - Show this help message
  clear   - Clear the terminal screen
  status  - Show current session and system status
  exit    - Exit the application

Sample Questions You Can Try:
  1. Attendance & Rules:
     - "What is the minimum attendance requirement to write exams?"
     - "What happens if my attendance is below 75%?"

  2. Course Syllabus:
     - "My department is Computer Science. What subjects are in Semester 5?"
     - "What topics are covered in the Artificial Intelligence subject?"

  3. Memory Test:
     - Step 1: "My department is Computer Science."
     - Step 2: "What subjects should I focus on this semester?"

  4. Campus Facilities & FAQs:
     - "What are the library working hours?"
     - "Where can I check my semester exam results?"

  5. System Tools:
     - "What is today's date and time?"
     - "I attended 32 out of 40 classes. Am I eligible for exams?"
-------------------------------------------------
"""


def clear_screen():
    """Clear the terminal console."""
    os.system("cls" if os.name == "nt" else "clear")


def check_api_keys() -> bool:
    """Validate that at least one LLM API key is present."""
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_groq = bool(os.getenv("GROQ_API_KEY"))
    has_google = bool(os.getenv("GOOGLE_API_KEY"))

    if not (has_openai or has_groq or has_google):
        print("\n" + "=" * 60)
        print(" [!] CONFIGURATION REQUIRED: Missing LLM API Key")
        print("=" * 60)
        print(" CampusMate AI requires an API key to run.")
        print("\n Quick Setup Steps:")
        print(" 1. Create a '.env' file in the 'campusmate-ai' folder.")
        print(" 2. Add one of the following API keys:")
        print("      OPENAI_API_KEY=your_key_here     (OpenAI)")
        print("      GROQ_API_KEY=your_key_here       (Free at console.groq.com)")
        print("      GOOGLE_API_KEY=your_key_here     (Free at aistudio.google.com)")
        print(" 3. Re-run: python main.py")
        print("=" * 60 + "\n")
        return False
    return True


def extract_text_from_response(content) -> str:
    """
    Safely extract pure text from various LLM response formats:
    - plain string
    - list of dicts/content blocks (e.g., [{'type': 'text', 'text': '...'}])
    - list of strings or nested message objects
    - LangChain AIMessage / BaseMessage objects
    - dict with 'text' or 'content' keys
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    # Handle LangChain BaseMessage / AIMessage
    if hasattr(content, "content"):
        return extract_text_from_response(content.content)

    # Handle dictionary representation
    if isinstance(content, dict):
        if content.get("type") == "text" and "text" in content:
            return str(content["text"]).strip()
        if "text" in content and isinstance(content["text"], str):
            return content["text"].strip()
        if "content" in content:
            return extract_text_from_response(content["content"])
        for key in ("text", "output", "result", "answer"):
            if key in content and content[key]:
                return extract_text_from_response(content[key])
        return ""

    # Handle list or tuple of content blocks
    if isinstance(content, (list, tuple)):
        extracted_parts = []
        for item in content:
            extracted = extract_text_from_response(item)
            if extracted:
                extracted_parts.append(extracted)
        return "\n".join(extracted_parts).strip()

    return str(content).strip()


def run_chatbot():
    """Main interactive terminal loop for CampusMate AI."""
    clear_screen()
    print(BANNER)

    if not check_api_keys():
        sys.exit(1)

    print("[Initializing CampusMate AI LangGraph Agent & RAG Vector Store...]")
    
    try:
        from langchain_core.messages import HumanMessage, AIMessage
        from agent import create_campusmate_graph
        from rag import get_vector_store

        # Initialize the LangGraph agent workflow
        app = create_campusmate_graph()
        
        # Pre-initialize embedding model and FAISS vector store ONCE during application startup
        get_vector_store()
        
        print("[System Ready. Start chatting below!]\n")
    except ImportError as ie:
        print(f"\n[!] Missing Dependencies: {ie}")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"\n[Error Initializing Agent]: {e}")
        print("Please check your dependencies and .env file.")
        sys.exit(1)

    # Unique thread ID to demonstrate LangGraph memory checkpointing
    session_thread_id = f"student_session_{uuid.uuid4().hex[:6]}"
    config = {"configurable": {"thread_id": session_thread_id}}

    while True:
        try:
            user_input = input("You: ").strip()

            # Handle empty input
            if not user_input:
                continue

            # Command: exit
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nThank you for using CampusMate AI. Have a great academic semester!\n")
                break

            # Command: clear
            if user_input.lower() == "clear":
                clear_screen()
                print(BANNER)
                continue

            # Command: help
            if user_input.lower() == "help":
                print(HELP_TEXT)
                continue

            # Command: status
            if user_input.lower() == "status":
                print(f"\n[Session Status]")
                print(f"- Thread ID: {session_thread_id}")
                print(f"- Memory: Active (LangGraph MemorySaver)")
                print(f"- Vector Store: FAISS (Local)\n")
                continue

            # Stream or invoke the LangGraph agent
            print("\nCampusMate AI is thinking...", end="\r", flush=True)

            input_data = {"messages": [HumanMessage(content=user_input)]}
            
            # Run graph invocation
            response_state = app.invoke(input_data, config=config)
            
            # Erase the thinking message
            print(" " * 40, end="\r", flush=True)

            # Extract the latest AI message
            messages = response_state.get("messages", [])
            final_response = None

            # Find last AI message with text content
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    extracted = extract_text_from_response(msg)
                    if extracted:
                        final_response = extracted
                        break

            if final_response:
                print(f"\nCampusMate AI: {final_response}\n")
            else:
                print("\nCampusMate AI: I processed your request, but could not formulate a text response.\n")

        except KeyboardInterrupt:
            print("\n\nSession ended by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n[An error occurred]: {e}\n")
            print("Please try asking another question or type 'help'.\n")


if __name__ == "__main__":
    run_chatbot()
