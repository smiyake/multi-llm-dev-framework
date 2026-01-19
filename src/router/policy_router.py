"""
Policy Router - Routing layer before LiteLLM

Routes tasks to appropriate models based on:
- Task type (sentiment, code_gen, etc.)
- GPU/CPU availability
- Cost optimization rules

Stage 1 (16GB VRAM): GPT-OSS-20B (Q4) unified for all local tasks
Stage 2 (96GB VRAM): Add Qwen3-32B for Japanese sentiment
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal, List
import httpx
import yaml
from pathlib import Path
from loguru import logger
from datetime import datetime
import json

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
    GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    GPU_AVAILABLE = False
    GPU_HANDLE = None
    logger.warning("NVIDIA GPU monitoring not available")

app = FastAPI(title="Policy Router", version="2.0.0")

# Configuration
CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "routing_rules.yaml"
PROMPT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "prompts.yaml"
LOG_DIR = Path(__file__).parent.parent.parent / "logs" / "routing"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Endpoints
LITELLM_URL = "http://127.0.0.1:4000/v1/chat/completions"
GPT_OSS_DIRECT = "http://127.0.0.1:8080/v1/chat/completions"  # llama.cpp server
VLLM_DIRECT = "http://127.0.0.1:8000/v1/chat/completions"

# Default model for 16GB environment
DEFAULT_MODEL = "gpt-oss-20b-q4"


class Message(BaseModel):
    role: str
    content: str


class RoutingRequest(BaseModel):
    messages: List[Message]
    task_type: Optional[Literal[
        "sentiment", "code_gen", "code_review",
        "log_summary", "document", "analysis",
        "architecture", "security_audit", "agent_support", "general"
    ]] = "general"
    max_tokens: int = 512
    temperature: float = 0.7
    force_model: Optional[str] = None
    language: Optional[str] = None  # For future Japanese routing


class RoutingResponse(BaseModel):
    model: str
    endpoint: str
    reason: str


class SentimentResult(BaseModel):
    """Standardized sentiment output schema for model-agnostic switching"""
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float  # 0.0 - 1.0
    rationale: str     # Brief explanation
    model_id: str      # For A/B testing


def load_routing_rules() -> dict:
    """Load routing rules from YAML config"""
    if CONFIG_PATH.exists():
        return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return {}


def load_prompt_config() -> dict:
    """Load prompt templates from YAML config"""
    if PROMPT_CONFIG_PATH.exists():
        return yaml.safe_load(PROMPT_CONFIG_PATH.read_text(encoding="utf-8"))
    return {}


def get_gpu_free_memory_mb() -> int:
    """Get GPU free memory in MB"""
    if not GPU_AVAILABLE:
        return 0
    try:
        info = pynvml.nvmlDeviceGetMemoryInfo(GPU_HANDLE)
        return info.free // (1024 * 1024)
    except Exception:
        return 0


def log_routing_decision(
    request_id: str,
    task_type: str,
    model: str,
    reason: str,
    latency_ms: Optional[float] = None,
    success: Optional[bool] = None
):
    """Log routing decisions for A/B evaluation"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "request_id": request_id,
        "task_type": task_type,
        "model": model,
        "reason": reason,
        "latency_ms": latency_ms,
        "success": success,
        "gpu_free_mb": get_gpu_free_memory_mb() if GPU_AVAILABLE else None
    }

    log_file = LOG_DIR / f"routing_{datetime.now().strftime('%Y%m%d')}.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    logger.debug(f"Logged routing: {log_entry}")


def choose_model(req: RoutingRequest) -> RoutingResponse:
    """
    Routing logic - GPT-OSS-20B unified for 16GB environment
    Returns: RoutingResponse with model, endpoint, and reason
    """
    # Force model override
    if req.force_model:
        return RoutingResponse(
            model=req.force_model,
            endpoint=LITELLM_URL,
            reason="Forced model selection"
        )

    # Load rules
    rules = load_routing_rules()
    task_rules = {r["task_type"]: r for r in rules.get("rules", [])}

    # Task type based routing
    if req.task_type in task_rules:
        rule = task_rules[req.task_type]
        model = rule["model"]

        # Determine endpoint
        if rule.get("endpoint") == "direct":
            endpoint = GPT_OSS_DIRECT
        elif rule.get("endpoint") == "litellm":
            endpoint = LITELLM_URL
        else:
            endpoint = GPT_OSS_DIRECT  # Default to local

        reason = rule.get("reason", "Task type rule")

        # GPU availability check
        free_mb = get_gpu_free_memory_mb()
        if free_mb > 0 and free_mb < 4000:  # Less than 4GB free
            logger.warning(f"GPU memory low ({free_mb}MB), using Q4 quantized")
            # Continue with Q4 model (already selected)
        elif not GPU_AVAILABLE and model != "claude-sonnet":
            # No GPU, fallback to cloud for non-Claude tasks
            logger.warning("No GPU available, falling back to Claude API")
            return RoutingResponse(
                model="claude-sonnet",
                endpoint=LITELLM_URL,
                reason="No GPU, fallback to cloud"
            )

        return RoutingResponse(model=model, endpoint=endpoint, reason=reason)

    # Default: GPT-OSS-20B for general tasks
    return RoutingResponse(
        model=DEFAULT_MODEL,
        endpoint=GPT_OSS_DIRECT,
        reason="Default routing to GPT-OSS-20B"
    )


@app.post("/v1/chat/completions")
async def route_completion(req: RoutingRequest):
    """Route chat completion request to appropriate model"""
    import uuid
    request_id = str(uuid.uuid4())[:8]
    start_time = datetime.now()

    routing = choose_model(req)
    logger.info(f"[{request_id}] Routing to {routing.model} via {routing.endpoint}: {routing.reason}")

    payload = {
        "model": routing.model,
        "messages": [m.model_dump() for m in req.messages],
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
    }

    async with httpx.AsyncClient(timeout=120) as client:
        try:
            r = await client.post(routing.endpoint, json=payload)
            r.raise_for_status()
            result = r.json()

            # Calculate latency
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000

            # Add routing info to response
            result["_routing"] = routing.model_dump()
            result["_routing"]["request_id"] = request_id
            result["_routing"]["latency_ms"] = latency_ms

            # Log for A/B evaluation
            log_routing_decision(
                request_id=request_id,
                task_type=req.task_type,
                model=routing.model,
                reason=routing.reason,
                latency_ms=latency_ms,
                success=True
            )

            return result

        except httpx.HTTPStatusError as e:
            logger.error(f"[{request_id}] Request failed: {e}")

            # Log failure
            log_routing_decision(
                request_id=request_id,
                task_type=req.task_type,
                model=routing.model,
                reason=routing.reason,
                success=False
            )

            # Fallback to Claude API
            if routing.model != "claude-sonnet":
                logger.info(f"[{request_id}] Falling back to Claude API")
                payload["model"] = "claude-sonnet"
                try:
                    r = await client.post(LITELLM_URL, json=payload)
                    result = r.json()
                    result["_routing"] = {
                        "model": "claude-sonnet",
                        "endpoint": LITELLM_URL,
                        "reason": "Fallback after local model failure",
                        "request_id": request_id
                    }
                    return result
                except Exception as fallback_error:
                    logger.error(f"[{request_id}] Fallback also failed: {fallback_error}")

            raise HTTPException(status_code=e.response.status_code, detail=str(e))

        except Exception as e:
            logger.error(f"[{request_id}] Unexpected error: {e}")
            log_routing_decision(
                request_id=request_id,
                task_type=req.task_type,
                model=routing.model,
                reason=routing.reason,
                success=False
            )
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/route")
async def get_routing(req: RoutingRequest) -> RoutingResponse:
    """Get routing decision without executing request"""
    return choose_model(req)


@app.get("/health")
async def health():
    """Health check endpoint"""
    gpu_free = get_gpu_free_memory_mb() if GPU_AVAILABLE else "N/A"
    return {
        "status": "ok",
        "gpu_available": GPU_AVAILABLE,
        "gpu_free_mb": gpu_free,
        "default_model": DEFAULT_MODEL,
        "stage": "1 (16GB VRAM)"
    }


@app.get("/models")
async def list_models():
    """List available models and their endpoints"""
    return {
        "models": [
            {"name": "gpt-oss-20b-q4", "endpoint": GPT_OSS_DIRECT, "type": "local-gpu", "vram": "~12GB"},
            {"name": "claude-sonnet", "endpoint": LITELLM_URL, "type": "cloud", "vram": "N/A"},
            {"name": "claude-haiku", "endpoint": LITELLM_URL, "type": "cloud", "vram": "N/A"},
        ],
        "future_models": [
            {"name": "qwen3-32b", "type": "local-gpu", "vram": "~24GB", "stage": "2 (96GB)"},
        ]
    }


@app.get("/logs/stats")
async def get_routing_stats():
    """Get routing statistics for A/B evaluation"""
    stats = {
        "total_requests": 0,
        "by_model": {},
        "by_task_type": {},
        "success_rate": 0.0,
        "avg_latency_ms": 0.0
    }

    # Read today's log
    log_file = LOG_DIR / f"routing_{datetime.now().strftime('%Y%m%d')}.jsonl"
    if log_file.exists():
        total_latency = 0.0
        success_count = 0

        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                stats["total_requests"] += 1

                model = entry.get("model", "unknown")
                task_type = entry.get("task_type", "unknown")

                stats["by_model"][model] = stats["by_model"].get(model, 0) + 1
                stats["by_task_type"][task_type] = stats["by_task_type"].get(task_type, 0) + 1

                if entry.get("latency_ms"):
                    total_latency += entry["latency_ms"]
                if entry.get("success"):
                    success_count += 1

        if stats["total_requests"] > 0:
            stats["success_rate"] = success_count / stats["total_requests"]
            stats["avg_latency_ms"] = total_latency / stats["total_requests"]

    return stats


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
