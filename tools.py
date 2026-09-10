"""
CampusMate AI - Tools Module
Defines LangChain tools callable by the AI Agent.
"""

from datetime import datetime
from langchain_core.tools import tool
from rag import retrieve_college_info


@tool
def get_current_datetime() -> str:
    """
    Get the current system date, day of the week, and local time.
    Use this tool whenever a student asks for today's date, current time, or day of the week.
    """
    now = datetime.now()
    formatted = now.strftime("%A, %B %d, %Y, %I:%M:%S %p")
    return f"Current System Date and Time: {formatted}"


@tool
def search_college_knowledge_base(query: str) -> str:
    """
    Search and retrieve official college documents, academic policies, attendance criteria,
    department syllabus, exam schedules, and FAQs from the college RAG vector database.
    
    Args:
        query: Specific search phrase or question about college rules, regulations, syllabus, or FAQs.
    """
    return retrieve_college_info(query)


@tool
def calculate_attendance_status(attended_classes: int, total_classes: int) -> str:
    """
    Calculate attendance percentage and evaluate eligibility against the 75% mandatory college regulation.
    
    Args:
        attended_classes: Number of classes the student has attended.
        total_classes: Total number of classes conducted.
    """
    if total_classes <= 0:
        return "Error: Total classes conducted must be greater than zero."
    if attended_classes < 0 or attended_classes > total_classes:
        return "Error: Attended classes cannot be negative or exceed total classes."
    
    percentage = (attended_classes / total_classes) * 100
    status = "ELIGIBLE" if percentage >= 75.0 else "SHORTAGE / NOT ELIGIBLE"
    
    needed_classes = 0
    if percentage < 75.0:
        # Calculate classes needed to reach 75%
        # (attended + x) / (total + x) >= 0.75 => x >= 3*total - 4*attended
        needed_classes = max(0, 3 * total_classes - 4 * attended_classes)

    summary = (
        f"Attendance Summary:\n"
        f"- Classes Attended: {attended_classes}/{total_classes}\n"
        f"- Attendance Percentage: {percentage:.2f}%\n"
        f"- Status: {status}\n"
        f"- Minimum College Requirement: 75.00%\n"
    )
    if status == "ELIGIBLE":
        summary += "✓ You meet the college attendance criteria for examinations."
    else:
        summary += (
            f"⚠ Warning: You have an attendance shortage. You need to attend the next "
            f"{needed_classes} consecutive classes without missing any to reach 75%."
        )
    return summary


# Export standard tools list
ALL_TOOLS = [
    get_current_datetime,
    search_college_knowledge_base,
    calculate_attendance_status,
]
