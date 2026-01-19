"""
Policy Router - Routing layer before LiteLLM

Routes tasks to appropriate models based on:
- Task type
- GPU/CPU load
- Cost optimization rules
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal, List
import httpx
import yaml
from pathlib import Path
from loguru import logger

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
    GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    GPU_AVAILABLE = False
    GPU_HANDLE = None
    logger.warning("NVIDIA GPU monitoring not available")

app = FastAPI(title="Policy Router", version="1.0.0")

# Configuration
CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "routing_rules.yaml"
LITELLM_URL = "http://127.0.0.1:4000/v1/chat/completions"
LLAMA_CPP_DIRECT = "http://127.0.0.1:8080/v1/chat/completions"
VLLM_DIRECT = "http://127.0.0.1:8000/v1/chat/completions"


class Message(BaseModel):
    role: str
    content: str


class RoutingRequest(BaseModel):
    messages: List[Message]
    task_type: Optional[Literal[
        "sentiment", "code_gen", "code_review",
        "log_summary", "document", "analysis",
        "architecture", "security_audit", "general"
    ]] = "general"
    max_tokens: int = 512
    temperature: float = 0.7
    force_model: Optional[str] = None


class RoutingResponse(BaseModel):
    model: str
    endpoint: str
    reason: str


def load_routing_rules() -> dict:
    """Load routing rules from YAML config"""
    if CONFIG_PATH.exists():
        return yaml.safe_load(CONFIG_PATH.read_text())
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


def choose_model(req: RoutingRequest) -> RoutingResponse:
    """
    Routing logic
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
        endpoint = LLAMA_CPP_DIRECT if rule.get("endpoint") == "direct" else LITELLM_URL
        reason = rule.get("reason", "Task type rule")

        # GPU availability check for vLLM
        if "vllm" in model:
            free_mb = get_gpu_free_memory_mb()
            if free_mb < 2000:  # Less than 2GB free
                logger.warning(f"GPU memory low ({free_mb}MB), falling back to llama.cpp")
                return RoutingResponse(
                    model="llama-cpp-7b",
                    endpoint=LLAMA_CPP_DIRECT,
                    reason=f"GPU memory low ({free_mb}MB), fallback to CPU"
                )

        return RoutingResponse(model=model, endpoint=endpoint, reason=reason)

    # Long context -> vLLM
    if req.max_tokens > 4096:
        return RoutingResponse(
            model="vllm-qwen-14b",
            endpoint=LITELLM_URL,
            reason="Long context benefits from vLLM"
        )

    # Default: vLLM
    return RoutingResponse(
        model="vllm-qwen-14b",
        endpoint=LITELLM_URL,
        reason="Default routing"
    )


@app.post("/v1/chat/completions")
async def route_completion(req: RoutingRequest):
    """Route chat completion request to appropriate model"""
    routing = choose_model(req)
    logger.info(f"Routing to {routing.model} via {routing.endpoint}: {routing.reason}")

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
            # Add routing info to response
            result["_routing"] = routing.model_dump()
            return result

        except httpx.HTTPStatusError as e:
            logger.error(f"Request failed: {e}")

            # Fallback chain
            if "vllm" in routing.model:
                logger.info("Falling back to llama.cpp")
                payload["model"] = "llama-cpp-7b"
                try:
                    r = await client.post(LLAMA_CPP_DIRECT, json=payload)
                    result = r.json()
                    result["_routing"] = {"model": "llama-cpp-7b", "endpoint": LLAMA_CPP_DIRECT, "reason": "Fallback"}
                    return result
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")

            raise HTTPException(status_code=e.response.status_code, detail=str(e))

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
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
    }


@app.get("/models")
async def list_models():
    """List available models and their endpoints"""
    return {
        "models": [
            {"name": "vllm-qwen-14b", "endpoint": VLLM_DIRECT, "type": "local-gpu"},
            {"name": "llama-cpp-7b", "endpoint": LLAMA_CPP_DIRECT, "type": "local-cpu"},
            {"name": "claude-sonnet", "endpoint": LITELLM_URL, "type": "cloud"},
            {"name": "claude-haiku", "endpoint": LITELLM_URL, "type": "cloud"},
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
