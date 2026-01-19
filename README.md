# Multi-LLM Development Framework

A generic framework for orchestrating multiple LLMs (Claude, Qwen, GPT-OSS, etc.) in a unified development workflow.

## Overview

This framework enables:
- **LLM Routing**: Route tasks to the most appropriate model based on task type, cost, and latency
- **Agent Coordination**: Coordinate multiple coding agents (Cline, Aider, OpenHands)
- **Resource Optimization**: Maximize GPU/CPU utilization across local and cloud LLMs
- **Cost Efficiency**: Minimize API costs by using local LLMs for bulk tasks

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Clients                               │
│  (Python scripts, Cline, Aider, OpenHands, etc.)            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Policy Router                             │
│  - Task type detection                                       │
│  - GPU/CPU load monitoring                                   │
│  - Cost optimization rules                                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      LiteLLM Proxy                           │
│  - Unified OpenAI-compatible API                             │
│  - Failover support                                          │
│  - Logging & metrics                                         │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
    ┌──────────┐        ┌──────────┐        ┌──────────┐
    │  vLLM    │        │llama.cpp │        │Claude API│
    │  (GPU)   │        │(CPU+GPU) │        │ (Cloud)  │
    └──────────┘        └──────────┘        └──────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- CUDA 12.x (for GPU inference)
- 16GB+ VRAM (for local models)

### Installation

```bash
# Clone
git clone https://github.com/your-org/multi-llm-dev-framework.git
cd multi-llm-dev-framework

# Setup
pip install -r requirements.txt

# Configure
cp config/litellm.example.yaml config/litellm.yaml
cp .env.example .env
# Edit .env with your API keys
```

### Start Services

```bash
# Start LiteLLM proxy
litellm --config config/litellm.yaml --port 4000

# Start Policy Router
uvicorn src.router.policy_router:app --host 0.0.0.0 --port 5000

# Optional: Start vLLM
vllm serve Qwen/Qwen2.5-14B-Instruct --port 8000

# Optional: Start llama.cpp server
./llama-server -m models/qwen2.5-7b-q4_k_m.gguf -c 8192 --port 8080
```

## Configuration

### LiteLLM Config

See `config/litellm.example.yaml` for model configuration.

### Policy Router Rules

Edit `config/routing_rules.yaml` to customize task routing:

```yaml
rules:
  - task_type: sentiment
    model: llama-cpp-7b
    reason: "Low latency, high throughput"

  - task_type: code_review
    model: claude-sonnet
    reason: "High quality review"
```

## Coding Agent Integration

### Supported Agents

| Agent | Use Case | Integration |
|-------|----------|-------------|
| **Cline** | IDE-based editing | VS Code extension, OpenAI-compatible API |
| **Aider** | CLI pair programming | Git diff-based, direct API |
| **OpenHands** | Autonomous coding | Overnight batch tasks |

### Workflow

```
1. Claude Code: Task planning & specification
       ↓
2. OpenHands: Bulk implementation (overnight)
       ↓
3. Aider: Refinement & diff-based changes
       ↓
4. Cline: Quick fixes in IDE
       ↓
5. Claude Code: Review & approval
```

## Project Structure

```
multi-llm-dev-framework/
├── config/
│   ├── litellm.example.yaml    # LiteLLM configuration template
│   └── routing_rules.yaml      # Policy Router rules
├── src/
│   ├── router/
│   │   └── policy_router.py    # FastAPI Policy Router
│   ├── agents/
│   │   └── agent_coordinator.py # Agent coordination logic
│   └── protocols/
│       └── llm_protocol.py     # LLM-to-LLM communication protocol
├── scripts/
│   ├── start_services.ps1      # Windows service starter
│   └── nightly.ps1             # Nightly job script
├── templates/
│   ├── task_card.md            # Task card template
│   └── handoff.md              # Agent handoff template
└── docs/
    └── architecture.md         # Detailed architecture docs
```

## LLM-to-LLM Protocol

For efficient inter-LLM communication, use JSON-based structured messages:

```json
{
  "msg_type": "task_assignment",
  "from": "orchestrator",
  "to": "implementer",
  "task_ref": "task-001",
  "context_refs": ["plan.md#section-2", "spec.md#api-design"],
  "instruction": "Implement API endpoint per spec",
  "constraints": ["Python 3.11+", "FastAPI"],
  "expected_output": "code_diff"
}
```

## Cost Optimization

| Task Type | Recommended Model | Cost | Latency |
|-----------|------------------|------|---------|
| Sentiment Analysis | Local 7B (llama.cpp) | $0 | Low |
| Code Generation | Local 14B (vLLM) | $0 | Medium |
| Code Review | Claude Sonnet | $$ | Medium |
| Architecture Design | Claude Opus | $$$ | High |

## Hardware Requirements

### Minimum (16GB VRAM)
- 7B model: Q4 quantization
- 14B model: Q4 quantization

### Recommended (96GB+ VRAM)
- 70B model: Q4/Q8 quantization
- Multiple models concurrent

## License

MIT License

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
