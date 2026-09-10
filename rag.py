"""
CampusMate AI - RAG (Retrieval-Augmented Generation) Module
Handles document loading, text chunking, FAISS vector store indexing, and document retrieval.
"""

import os
import sys
import warnings
import logging
from typing import List, Optional

# ----------------------------------------------------------------------
# Suppress HuggingFace, Transformers, and library warnings/progress bars
# ----------------------------------------------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["DATASETS_VERBOSITY"] = "error"

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
]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

try:
    import transformers
    transformers.logging.set_verbosity_error()
    transformers.utils.logging.disable_progress_bar()
except Exception:
    pass

try:
    import huggingface_hub
    huggingface_hub.logging.set_verbosity_error()
except Exception:
    pass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Sample documents directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

SAMPLE_FILES = {
    "college_regulations.txt": """================================================================================
COLLEGE ACADEMIC REGULATIONS & POLICIES (DEMO DATA)
Note: This is sample demonstration data for CampusMate AI and can be replaced with actual institution documents.
================================================================================

1. ATTENDANCE REQUIREMENTS:
- Minimum mandatory attendance is 75 percent across all registered theory and laboratory courses.
- Students with attendance between 65% and 74% due to verified medical emergencies or institutional duty representation may apply for condonation subject to Dean/Principal approval.
- Students below minimum attendance (less than 75% without approved condonation) will NOT be eligible to appear for the end-semester examinations and must redo the course in subsequent semesters.
- Attendance status is updated weekly on the student portal.

2. EXAMINATION & ASSESSMENT REGULATIONS:
- Internal continuous assessments account for 40% of the overall course grade, and end-semester examinations account for 60%.
- Examination application forms must be submitted through the student portal strictly before the notified deadlines.
- Late submission of examination applications attracts a penalty fee of Rs. 500 up to 3 days after the deadline; no applications are accepted after the grace period.
- Hall tickets will be issued only to candidates with cleared dues and valid attendance eligibility.

3. ACADEMIC CALENDAR & COURSE REGISTRATION:
- Students should strictly follow the academic calendar issued at the beginning of each semester.
- Course registration / add-drop period ends within the first two weeks of the semester commencement.
- A minimum CGPA of 5.0 is required for the award of undergraduate degree, and no active backlogs must exist at the time of graduation.

4. CODE OF CONDUCT & DISCIPLINARY POLICY:
- Possession or use of unauthorized electronic devices during examinations is treated as academic malpractice and referred to the Disciplinary Committee.
- Ragging in any form inside or outside the campus premises is strictly prohibited with zero tolerance and attracts legal action under state regulations.
""",
    "faq.txt": """================================================================================
CAMPUS FREQUENTLY ASKED QUESTIONS (FAQ) (DEMO DATA)
Note: This is sample demonstration data for CampusMate AI and can be replaced with actual institution documents.
================================================================================

Q: What are the central library working hours?
A: Central Library working hours are 9:00 AM to 6:00 PM on all official working days (Monday through Saturday). The digital library and reading room remain open until 8:00 PM during examination weeks.

Q: Who should students contact for academic issues, marks discrepancies, or course advisement?
A: Students should contact their respective Academic Department / Head of Department (HOD) or designated Faculty Mentor for academic issues, course syllabus queries, or internal marks clarifications.

Q: Where and how can students view examination results?
A: Examination results are officially published and available through the online Student Information Portal (ERP) using student registration number and date of birth credentials.

Q: How can students apply for leave or on-duty (OD) permission?
A: Leave and On-Duty (OD) applications must be submitted via the online ERP portal at least 2 days in advance, supported by relevant event certificates or medical records approved by the Class Advisor and HOD.

Q: What are the college canteen and dining facility hours?
A: The campus canteen is open from 7:30 AM to 7:00 PM. Breakfast is served from 7:30 AM to 9:30 AM, Lunch from 12:00 PM to 2:00 PM, and Snacks/Refreshments from 3:30 PM to 6:30 PM.

Q: How do I obtain bonafide certificates or transcript copies?
A: Bonafide requests can be applied online through the Student Portal or submitted in person at the Academic Administrative Section with 2 working days turnaround time.
""",
    "syllabus.txt": """================================================================================
ACADEMIC COURSE SYLLABUS & CURRICULUM (DEMO DATA)
Note: This is sample demonstration data for CampusMate AI and can be replaced with actual institution documents.
================================================================================

DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING (CSE)

Semester 5 Curriculum:
Course Code | Subject Name | Credits | Core Topics Covered
--------------------------------------------------------------------------------
CS501 | Artificial Intelligence | 4 Credits | Search Algorithms, Knowledge Representation, Heuristic Search, Propositional Logic, Expert Systems, Game Playing.
CS502 | Database Management Systems | 4 Credits | Relational Algebra, SQL, Normalization (1NF to BCNF), Transaction Processing, Concurrency Control, Indexing.
CS503 | Computer Networks | 3 Credits | OSI & TCP/IP Reference Models, Data Link Protocols, Routing Algorithms, TCP/UDP Transport Layer, Network Security Protocols.
CS504 | Software Engineering | 3 Credits | Agile Methodologies, Software Requirement Specifications (SRS), Design Patterns, Software Testing, DevOps lifecycle.
CS505 | Machine Learning Fundamentals | 4 Credits | Supervised Learning, Unsupervised Clustering, Regression, Decision Trees, Neural Network Basics, Model Evaluation Metrics.
CS506 | Database Management Systems Lab | 2 Credits | Practical SQL Queries, PL/SQL Procedures, Triggers, End-to-End Database Schema Application Project.
CS507 | Artificial Intelligence & Machine Learning Lab | 2 Credits | Python implementations of Search Algorithms, Scikit-Learn Regression & Classification, Neural Net models.

Semester 6 Curriculum:
- CS601: Cloud Computing & Distributed Systems
- CS602: Cryptography and Network Security
- CS603: Web Technologies & Full Stack Development
- CS604: Deep Learning & NLP
- CS605: Elective I (Internet of Things / Blockchain Technology)

DEPARTMENT OF INFORMATION TECHNOLOGY (IT)
Semester 5 Subjects:
- IT501: Web Frameworks & Microservices
- IT502: Design & Analysis of Algorithms
- IT503: Information Security
- IT504: Cloud Infrastructure & Virtualization

DEPARTMENT OF ELECTRONICS & COMMUNICATION ENGINEERING (ECE)
Semester 5 Subjects:
- EC501: Digital Signal Processing
- EC502: Microprocessors & Microcontrollers
- EC503: VLSI Design
- EC504: Transmission Lines and Waveguides
"""
}

# Global singletons for embeddings model and vector store
_EMBEDDINGS = None
_VECTOR_STORE = None


def ensure_sample_data():
    """Ensure data directory and sample text files exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    for filename, content in SAMPLE_FILES.items():
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)


def get_embeddings():
    """
    Initialize or return cached embeddings model singleton.
    Loaded ONLY ONCE upon startup and reused across all subsequent queries.
    """
    global _EMBEDDINGS
    if _EMBEDDINGS is not None:
        return _EMBEDDINGS

    # 1. Try HuggingFace Embeddings (local, no API cost)
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        _EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return _EMBEDDINGS
    except Exception:
        pass

    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        _EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return _EMBEDDINGS
    except Exception:
        pass

    # 2. Try OpenAI Embeddings if OPENAI_API_KEY exists
    if os.getenv("OPENAI_API_KEY"):
        try:
            from langchain_openai import OpenAIEmbeddings
            _EMBEDDINGS = OpenAIEmbeddings()
            return _EMBEDDINGS
        except Exception:
            pass

    # 3. Fallback to lightweight embeddings
    try:
        from langchain_community.embeddings import FakeEmbeddings
        _EMBEDDINGS = FakeEmbeddings(size=384)
        return _EMBEDDINGS
    except Exception:
        raise RuntimeError("No suitable embedding model could be loaded.")


def load_and_split_documents() -> List[Document]:
    """Load all text documents from data directory and split into chunks."""
    ensure_sample_data()
    documents = []

    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt"):
            filepath = os.path.join(DATA_DIR, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
                documents.append(Document(page_content=text, metadata={"source": filename}))
            except Exception:
                pass

    # Split documents into chunks for optimal retrieval
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def get_vector_store() -> FAISS:
    """
    Initialize or return cached FAISS vector store singleton.
    Created/indexed ONLY ONCE and reused for every user question.
    """
    global _VECTOR_STORE
    if _VECTOR_STORE is None:
        chunks = load_and_split_documents()
        embeddings = get_embeddings()
        _VECTOR_STORE = FAISS.from_documents(chunks, embeddings)
    return _VECTOR_STORE


def retrieve_college_info(query: str, k: int = 3) -> str:
    """
    Retrieve relevant college information for a given query.
    Returns formatted context string.
    """
    try:
        vector_store = get_vector_store()
        docs = vector_store.similarity_search(query, k=k)
        if not docs:
            return "No specific college records found matching the query in the knowledge base."
        
        results = []
        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "Campus Documentation")
            results.append(f"--- Document Source [{source}] (Chunk {i}) ---\n{doc.page_content.strip()}")
        
        return "\n\n".join(results)
    except Exception as e:
        return f"Error retrieving college information from vector store: {str(e)}"
