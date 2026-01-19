# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Multi-LLM Development Framework - A generic framework for orchestrating multiple LLMs (Claude, Qwen, GPT-OSS, etc.) in unified development workflows.

## Architecture

```
Clients (Python, Cline, Aider, OpenHands)
         │
         ▼
    Policy Router (FastAPI) - Task-based routing
         │
         ▼
    LiteLLM Proxy - Unified OpenAI-compatible API
         │
    ┌────┼────┐
    ▼    ▼    ▼
  vLLM  llama.cpp  Claude API
```

## Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| Policy Router | `src/router/policy_router.py` | Route tasks to appropriate models |
| LLM Protocol | `src/protocols/llm_protocol.py` | LLM-to-LLM communication |
| Agent Coordinator | `src/agents/agent_coordinator.py` | Manage coding agents |
| Config | `config/` | LiteLLM and routing rules |

## Commands

```bash
# Start services
.\scripts\start_services.ps1 -All

# Run Policy Router only
uvicorn src.router.policy_router:app --port 5000

# Test
pytest tests/
```

## Routing Rules

| Task Type | Model | Reason |
|-----------|-------|--------|
| `sentiment` | llama-cpp-7b | Low latency |
| `code_gen` | vllm-qwen-14b | High throughput |
| `code_review` | claude-sonnet | High quality |
| `architecture` | claude-sonnet | Complex reasoning |

## File Structure

```
.multi-llm/           # Working directory for multi-LLM coordination
├── tasks/            # Task queue (pending/in-progress/completed)
├── plans/            # Planning documents
├── code/             # Generated code
├── reviews/          # Review results
└── shared/           # Status and shared data
```

## Development Guidelines

1. **Task-based routing**: Use `task_type` parameter to route to appropriate model
2. **Context references**: Use ContextRef instead of re-sending full content
3. **Structured handoffs**: Use Handoff protocol for agent transitions
4. **Cost optimization**: Prefer local models for bulk tasks

## Integration with Other Projects

This framework is designed to be used across multiple projects. To integrate:

1. Copy `.multi-llm/` directory structure to your project
2. Configure `config/litellm.yaml` for your models
3. Start services with `start_services.ps1`
4. Use Policy Router API at `http://localhost:5000`
