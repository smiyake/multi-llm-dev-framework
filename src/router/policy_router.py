"""
Policy Router - Routing layer before LiteLLM

Routes tasks to appropriate models based on:
- Task type (sentiment, code_gen, etc.)
- GPU/CPU availability
- Cost optimization rules

Stage 1 (16GB VRAM): GPT-OSS-20B (Q4) unified for all local tasks
Stage 2 (96GB VRAM): Add Qwen3-32B for Japanese sentiment

Configuration:
    Environment variables (highest priority):
        POLICY_ROUTER_PORT: Port to listen on (default: 5000)
        POLICY_ROUTER_HOST: Host to bind to (default: 0.0.0.0)
        LITELLM_URL: LiteLLM endpoint (default: http://127.0.0.1:4000/v1/chat/completions)
        OLLAMA_URL: Ollama endpoint (default: http://127.0.0.1:11434/v1/chat/completions)
        VLLM_URL: vLLM endpoint (default: http://127.0.0.1:8000/v1/chat/completions)
        PORTS_CONFIG_PATH: Path to ports.yaml config file

    Config file (ports.yaml):
        Searched in: current dir, ./config/, PORTS_CONFIG_PATH env var
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal, List, Tuple
import httpx
import os
import yaml
from pathlib import Path
from loguru import logger
from datetime import datetime
import json
import sys

# Try to import port-registry (optional dependency)
try:
    from port_registry import ServiceRegistry
    PORT_REGISTRY_AVAILABLE = True
except ImportError:
    PORT_REGISTRY_AVAILABLE = False
    logger.debug("port-registry not installed, using config file only")

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
    GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    GPU_AVAILABLE = False
    GPU_HANDLE = None
    logger.warning("NVIDIA GPU monitoring not available")

app = FastAPI(title="Policy Router", version="2.1.0")

# Configuration paths (relative to this package)
_PACKAGE_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = _PACKAGE_ROOT / "config" / "routing_rules.yaml"
PROMPT_CONFIG_PATH = _PACKAGE_ROOT / "config" / "prompts.yaml"
LOG_DIR = _PACKAGE_ROOT / "logs" / "routing"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def find_ports_config() -> Optional[Path]:
    """Find ports.yaml config file in standard locations."""
    # 1. Environment variable (highest priority)
    env_path = os.environ.get("PORTS_CONFIG_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    # 2. Search paths (relative to cwd and package)
    search_paths = [
        Path.cwd() / "config" / "ports.yaml",
        Path.cwd() / "ports.yaml",
        _PACKAGE_ROOT / "config" / "ports.yaml",
    ]

    for path in search_paths:
        if path.exists():
            return path

    return None


def load_ports_config() -> dict:
    """Load port configuration from ports.yaml."""
    config_path = find_ports_config()
    if config_path:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        logger.info(f"Loaded ports config from {config_path}")
        return config

    logger.debug("No ports.yaml found, using defaults/env vars")
    return {}


def get_endpoint(service_name: str, env_var: str, default_port: int, path: str = "/v1/chat/completions") -> str:
    """Get endpoint URL from environment variable or config.

    Priority: environment variable > ports.yaml > default
    """
    # 1. Environment variable (highest priority)
    env_url = os.environ.get(env_var)
    if env_url:
        return env_url

    # 2. Config file
    config = load_ports_config()
    services = config.get("services", {})
    service = services.get(service_name, {})

    if service:
        host = service.get("host", "127.0.0.1")
        port = service.get("port", default_port)
        return f"http://{host}:{port}{path}"

    # 3. Default
    return f"http://127.0.0.1:{default_port}{path}"


# Load endpoints (environment variables > config > defaults)
LITELLM_URL = get_endpoint("litellm", "LITELLM_URL", 4000)
OLLAMA_URL = get_endpoint("ollama", "OLLAMA_URL", 11434)
GPT_OSS_DIRECT = OLLAMA_URL  # Redirect to Ollama
VLLM_DIRECT = get_endpoint("vllm", "VLLM_URL", 8000)

# Get default model from config or environment
_ports_config = load_ports_config()
_models_config = _ports_config.get("models", {})
_default_backend = _ports_config.get("default_backend", "ollama")
DEFAULT_MODEL = os.environ.get(
    "DEFAULT_MODEL",
    _models_config.get("qwen3-coder", {}).get("model_name", "qwen3-coder:30b")
)

logger.info(f"Endpoints configured: LiteLLM={LITELLM_URL}, Ollama={OLLAMA_URL}, vLLM={VLLM_DIRECT}")
logger.info(f"Default model: {DEFAULT_MODEL} (backend: {_default_backend})")


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


def get_server_config() -> Tuple[str, int]:
    """Get server host and port from environment or config.

    Priority: environment variable > ports.yaml > default
    """
    # Environment variables (highest priority)
    host = os.environ.get("POLICY_ROUTER_HOST")
    port = os.environ.get("POLICY_ROUTER_PORT")

    if host and port:
        return host, int(port)

    # Config file
    config = load_ports_config()
    service_config = config.get("services", {}).get("policy_router", {})

    return (
        host or service_config.get("host", "0.0.0.0"),
        int(port) if port else service_config.get("port", 5000)
    )


@app.on_event("startup")
async def register_with_port_registry():
    """Register Policy Router with port-registry on startup (if available)."""
    if not PORT_REGISTRY_AVAILABLE:
        return

    try:
        # Use state_dir from environment or default
        state_dir = Path(os.environ.get(
            "PORT_REGISTRY_STATE_DIR",
            _PACKAGE_ROOT / "config" / "ports"
        ))
        state_dir.mkdir(parents=True, exist_ok=True)

        registry = ServiceRegistry(
            service_name="policy-router",
            state_dir=state_dir,
            cmdline_patterns=["policy_router", "uvicorn"],
        )
        host, port = get_server_config()
        registry.on_start(port=port, host=host)
        logger.info(f"Registered with port-registry: {host}:{port}")
    except Exception as e:
        logger.warning(f"Failed to register with port-registry: {e}")


if __name__ == "__main__":
    import uvicorn
    host, port = get_server_config()
    logger.info(f"Starting Policy Router on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
