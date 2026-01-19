"""
LLM-to-LLM Communication Protocol

Efficient structured communication between LLMs to:
- Reduce token consumption
- Enable context references without re-sending
- Support parallel agent coordination
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime
from enum import Enum
import json


class MessageType(str, Enum):
    """Types of inter-LLM messages"""
    TASK_ASSIGNMENT = "task_assignment"
    TASK_RESULT = "task_result"
    CONTEXT_UPDATE = "context_update"
    REVIEW_REQUEST = "review_request"
    REVIEW_RESULT = "review_result"
    HANDOFF = "handoff"
    STATUS_UPDATE = "status_update"


class OutputFormat(str, Enum):
    """Expected output formats"""
    CODE_DIFF = "code_diff"
    CODE_FILE = "code_file"
    MARKDOWN = "markdown"
    JSON = "json"
    PLAIN_TEXT = "plain_text"


class ContextRef(BaseModel):
    """Reference to existing context without re-sending"""
    file_path: str
    section: Optional[str] = None  # e.g., "## API Design"
    line_range: Optional[tuple[int, int]] = None  # e.g., (10, 50)


class LLMMessage(BaseModel):
    """Base message for LLM-to-LLM communication"""
    msg_type: MessageType
    msg_id: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S%f"))
    from_agent: str
    to_agent: str
    timestamp: datetime = Field(default_factory=datetime.now)


class TaskAssignment(LLMMessage):
    """Task assignment from orchestrator to implementer"""
    msg_type: MessageType = MessageType.TASK_ASSIGNMENT
    task_ref: str
    context_refs: List[ContextRef] = []  # References instead of full content
    instruction: str  # Brief, specific instruction
    constraints: List[str] = []
    expected_output: OutputFormat = OutputFormat.CODE_DIFF


class TaskResult(LLMMessage):
    """Task result from implementer to orchestrator"""
    msg_type: MessageType = MessageType.TASK_RESULT
    task_ref: str
    status: Literal["success", "partial", "failed"]
    output_ref: Optional[ContextRef] = None  # Where output is stored
    summary: str  # Brief summary (< 100 tokens)
    issues: List[str] = []


class ReviewRequest(LLMMessage):
    """Review request"""
    msg_type: MessageType = MessageType.REVIEW_REQUEST
    code_ref: ContextRef
    review_focus: List[str] = []  # e.g., ["security", "performance"]
    max_issues: int = 5


class ReviewResult(LLMMessage):
    """Review result"""
    msg_type: MessageType = MessageType.REVIEW_RESULT
    review_ref: str
    verdict: Literal["approve", "request_changes", "comment"]
    issues: List[dict] = []  # [{"severity": "high", "line": 42, "message": "..."}]
    summary: str


class Handoff(LLMMessage):
    """Agent handoff message"""
    msg_type: MessageType = MessageType.HANDOFF
    completed_work: List[str]  # Brief list of completed items
    next_action: str  # What the next agent should do
    context_refs: List[ContextRef] = []
    validation_method: Optional[str] = None


class StatusUpdate(LLMMessage):
    """Status update for monitoring"""
    msg_type: MessageType = MessageType.STATUS_UPDATE
    status: Literal["idle", "busy", "error"]
    current_task: Optional[str] = None
    progress_percent: Optional[int] = None
    eta_seconds: Optional[int] = None


# Utility functions
def serialize_message(msg: LLMMessage) -> str:
    """Serialize message to JSON string"""
    return msg.model_dump_json(indent=2)


def deserialize_message(json_str: str) -> LLMMessage:
    """Deserialize JSON string to appropriate message type"""
    data = json.loads(json_str)
    msg_type = data.get("msg_type")

    type_map = {
        MessageType.TASK_ASSIGNMENT.value: TaskAssignment,
        MessageType.TASK_RESULT.value: TaskResult,
        MessageType.REVIEW_REQUEST.value: ReviewRequest,
        MessageType.REVIEW_RESULT.value: ReviewResult,
        MessageType.HANDOFF.value: Handoff,
        MessageType.STATUS_UPDATE.value: StatusUpdate,
    }

    cls = type_map.get(msg_type, LLMMessage)
    return cls(**data)


# Example usage
if __name__ == "__main__":
    # Create a task assignment
    task = TaskAssignment(
        from_agent="orchestrator",
        to_agent="implementer-qwen",
        task_ref="task-001",
        context_refs=[
            ContextRef(file_path="docs/spec.md", section="## API Design"),
            ContextRef(file_path="src/api/routes.py", line_range=(1, 50)),
        ],
        instruction="Implement the user authentication endpoint per spec",
        constraints=["Python 3.11+", "FastAPI", "No external auth libraries"],
        expected_output=OutputFormat.CODE_DIFF,
    )

    print("=== Task Assignment ===")
    print(serialize_message(task))

    # Create a task result
    result = TaskResult(
        from_agent="implementer-qwen",
        to_agent="orchestrator",
        task_ref="task-001",
        status="success",
        output_ref=ContextRef(file_path=".multi-llm/code/qwen/auth_endpoint.py"),
        summary="Implemented /auth/login endpoint with JWT token generation",
        issues=[],
    )

    print("\n=== Task Result ===")
    print(serialize_message(result))

    # Create a handoff
    handoff = Handoff(
        from_agent="implementer-qwen",
        to_agent="reviewer-gpt",
        completed_work=[
            "Auth endpoint implementation",
            "Unit tests added",
            "Error handling for invalid credentials",
        ],
        next_action="Review code for security issues",
        context_refs=[
            ContextRef(file_path=".multi-llm/code/qwen/auth_endpoint.py"),
            ContextRef(file_path=".multi-llm/code/qwen/test_auth.py"),
        ],
        validation_method="pytest tests/test_auth.py",
    )

    print("\n=== Handoff ===")
    print(serialize_message(handoff))
