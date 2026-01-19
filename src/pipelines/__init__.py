"""
Multi-LLM Pipelines

Standardized pipelines for common tasks with model-agnostic output schemas.
"""
from .sentiment_pipeline import (
    SentimentPipeline,
    SentimentInput,
    SentimentResult,
    analyze_sentiment,
)

__all__ = [
    "SentimentPipeline",
    "SentimentInput",
    "SentimentResult",
    "analyze_sentiment",
]
