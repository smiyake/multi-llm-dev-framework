"""
Agent Coordinator

Coordinates multiple coding agents (Cline, Aider, OpenHands)
with proper task handoff and status tracking.
"""
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import json
from loguru import logger
from pydantic import BaseModel


class AgentType(str, Enum):
    """Supported coding agents"""
    CLAUDE_CODE = "claude-code"
    CLINE = "cline"
    AIDER = "aider"
    OPENHANDS = "openhands"


class AgentStatus(str, Enum):
    """Agent status"""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"


class TaskStatus(str, Enum):
    """Task status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class Task(BaseModel):
    """Task definition"""
    task_id: str
    title: str
    description: str
    assigned_to: Optional[AgentType] = None
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 5  # 1=highest, 10=lowest
    inputs: Dict[str, Any] = {}
    outputs: Dict[str, Any] = {}
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()


class AgentInfo(BaseModel):
    """Agent information"""
    agent_type: AgentType
    status: AgentStatus = AgentStatus.IDLE
    current_task: Optional[str] = None
    last_heartbeat: datetime = datetime.now()
    capabilities: List[str] = []


class AgentCoordinator:
    """
    Coordinates multiple coding agents.

    Workflow:
    1. Claude Code: Task planning & specification
    2. OpenHands: Bulk implementation (overnight)
    3. Aider: Refinement & diff-based changes
    4. Cline: Quick fixes in IDE
    5. Claude Code: Review & approval
    """

    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir
        self.multi_llm_dir = workspace_dir / ".multi-llm"
        self._init_directories()
        self.agents: Dict[AgentType, AgentInfo] = {}
        self._init_agents()

    def _init_directories(self):
        """Initialize .multi-llm directory structure"""
        dirs = [
            self.multi_llm_dir / "plans",
            self.multi_llm_dir / "tasks" / "pending",
            self.multi_llm_dir / "tasks" / "in-progress",
            self.multi_llm_dir / "tasks" / "completed",
            self.multi_llm_dir / "code",
            self.multi_llm_dir / "reviews",
            self.multi_llm_dir / "shared",
            self.multi_llm_dir / "logs",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def _init_agents(self):
        """Initialize agent registry"""
        self.agents = {
            AgentType.CLAUDE_CODE: AgentInfo(
                agent_type=AgentType.CLAUDE_CODE,
                capabilities=["planning", "review", "debugging", "architecture"]
            ),
            AgentType.CLINE: AgentInfo(
                agent_type=AgentType.CLINE,
                capabilities=["quick_edit", "ide_integration", "small_changes"]
            ),
            AgentType.AIDER: AgentInfo(
                agent_type=AgentType.AIDER,
                capabilities=["refactoring", "git_diff", "batch_changes", "testing"]
            ),
            AgentType.OPENHANDS: AgentInfo(
                agent_type=AgentType.OPENHANDS,
                capabilities=["autonomous", "overnight", "large_tasks", "exploration"]
            ),
        }

    def create_task(self, title: str, description: str, priority: int = 5, **inputs) -> Task:
        """Create a new task"""
        task_id = f"task-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        task = Task(
            task_id=task_id,
            title=title,
            description=description,
            priority=priority,
            inputs=inputs,
        )

        # Save to pending
        task_file = self.multi_llm_dir / "tasks" / "pending" / f"{task_id}.json"
        task_file.write_text(task.model_dump_json(indent=2))

        logger.info(f"Task created: {task_id} - {title}")
        return task

    def assign_task(self, task_id: str, agent_type: AgentType) -> bool:
        """Assign task to an agent"""
        pending_dir = self.multi_llm_dir / "tasks" / "pending"
        in_progress_dir = self.multi_llm_dir / "tasks" / "in-progress"

        task_file = pending_dir / f"{task_id}.json"
        if not task_file.exists():
            logger.error(f"Task not found: {task_id}")
            return False

        task = Task(**json.loads(task_file.read_text()))
        task.assigned_to = agent_type
        task.status = TaskStatus.IN_PROGRESS
        task.updated_at = datetime.now()

        # Move to in-progress
        new_path = in_progress_dir / f"{task_id}.json"
        new_path.write_text(task.model_dump_json(indent=2))
        task_file.unlink()

        # Update agent status
        self.agents[agent_type].status = AgentStatus.BUSY
        self.agents[agent_type].current_task = task_id
        self.agents[agent_type].last_heartbeat = datetime.now()

        logger.info(f"Task {task_id} assigned to {agent_type}")
        return True

    def complete_task(self, task_id: str, outputs: Dict[str, Any] = None) -> bool:
        """Mark task as completed"""
        in_progress_dir = self.multi_llm_dir / "tasks" / "in-progress"
        completed_dir = self.multi_llm_dir / "tasks" / "completed"

        task_file = in_progress_dir / f"{task_id}.json"
        if not task_file.exists():
            logger.error(f"Task not found in progress: {task_id}")
            return False

        task = Task(**json.loads(task_file.read_text()))
        task.status = TaskStatus.COMPLETED
        task.outputs = outputs or {}
        task.updated_at = datetime.now()

        # Move to completed
        new_path = completed_dir / f"{task_id}.json"
        new_path.write_text(task.model_dump_json(indent=2))
        task_file.unlink()

        # Update agent status
        if task.assigned_to:
            self.agents[task.assigned_to].status = AgentStatus.IDLE
            self.agents[task.assigned_to].current_task = None
            self.agents[task.assigned_to].last_heartbeat = datetime.now()

        logger.success(f"Task completed: {task_id}")
        return True

    def get_pending_tasks(self) -> List[Task]:
        """Get all pending tasks"""
        pending_dir = self.multi_llm_dir / "tasks" / "pending"
        tasks = []
        for f in pending_dir.glob("*.json"):
            task = Task(**json.loads(f.read_text()))
            tasks.append(task)
        return sorted(tasks, key=lambda t: t.priority)

    def get_agent_for_task(self, task: Task) -> AgentType:
        """
        Determine best agent for a task.

        Rules:
        - Planning/architecture -> Claude Code
        - Large autonomous tasks -> OpenHands
        - Git diff-based changes -> Aider
        - Quick IDE edits -> Cline
        """
        desc_lower = task.description.lower()

        # Planning tasks
        if any(k in desc_lower for k in ["plan", "design", "architecture", "review"]):
            return AgentType.CLAUDE_CODE

        # Large autonomous tasks
        if any(k in desc_lower for k in ["implement", "overnight", "autonomous", "bulk"]):
            return AgentType.OPENHANDS

        # Refactoring/batch changes
        if any(k in desc_lower for k in ["refactor", "batch", "multiple files", "test"]):
            return AgentType.AIDER

        # Quick edits
        if any(k in desc_lower for k in ["fix", "quick", "small", "typo"]):
            return AgentType.CLINE

        # Default to Aider for implementation
        return AgentType.AIDER

    def update_status(self) -> Dict[str, Any]:
        """Update and return system status"""
        status = {
            "agents": {a.agent_type: a.model_dump() for a in self.agents.values()},
            "tasks": {
                "pending": len(list((self.multi_llm_dir / "tasks" / "pending").glob("*.json"))),
                "in_progress": len(list((self.multi_llm_dir / "tasks" / "in-progress").glob("*.json"))),
                "completed": len(list((self.multi_llm_dir / "tasks" / "completed").glob("*.json"))),
            },
            "updated_at": datetime.now().isoformat(),
        }

        # Save to shared
        status_file = self.multi_llm_dir / "shared" / "status.json"
        status_file.write_text(json.dumps(status, indent=2, default=str))

        return status


# Example usage
if __name__ == "__main__":
    # Initialize coordinator
    coordinator = AgentCoordinator(Path("."))

    # Create a task
    task = coordinator.create_task(
        title="Implement user authentication",
        description="Implement JWT-based authentication for the API",
        priority=2,
        plan_file=".multi-llm/plans/auth-plan.md",
    )

    # Determine best agent
    best_agent = coordinator.get_agent_for_task(task)
    print(f"Best agent for task: {best_agent}")

    # Assign task
    coordinator.assign_task(task.task_id, best_agent)

    # Update status
    status = coordinator.update_status()
    print(json.dumps(status, indent=2, default=str))
