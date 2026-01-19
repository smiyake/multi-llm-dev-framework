"""
Sentiment Analysis Pipeline

Standardized sentiment analysis with model-agnostic output schema.
Supports GPT-OSS-20B (Stage 1) and future Qwen3-32B (Stage 2).

Output Schema:
{
    "sentiment": "positive" | "negative" | "neutral",
    "confidence": 0.0 - 1.0,
    "rationale": "brief explanation",
    "model_id": "gpt-oss-20b-q4"
}
"""
import httpx
import json
import re
from typing import Optional, Literal
from pydantic import BaseModel, Field, validator
from pathlib import Path
from loguru import logger
import yaml

# Configuration
CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "prompts.yaml"
ROUTER_URL = "http://127.0.0.1:5000"


class SentimentInput(BaseModel):
    """Input for sentiment analysis"""
    text: str
    language: Optional[str] = None  # auto-detect if None
    source: Optional[str] = None    # "news", "social", "report"


class SentimentResult(BaseModel):
    """Standardized sentiment output schema for model-agnostic switching"""
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    model_id: str

    @validator("confidence")
    def round_confidence(cls, v):
        return round(v, 3)


class SentimentPipeline:
    """
    Sentiment analysis pipeline with standardized output.

    Features:
    - Model-agnostic: Works with GPT-OSS-20B, Qwen3, etc.
    - Language detection (future: route Japanese to Qwen3)
    - Confidence thresholding
    - A/B evaluation logging
    """

    def __init__(self, router_url: str = ROUTER_URL):
        self.router_url = router_url
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> dict:
        """Load prompt templates from config"""
        if CONFIG_PATH.exists():
            return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        return self._default_prompts()

    def _default_prompts(self) -> dict:
        """Default prompts if config not found"""
        return {
            "sentiment": {
                "system": """You are a financial sentiment analyzer. Analyze the given text and respond with ONLY a JSON object in this exact format:
{"sentiment": "positive" or "negative" or "neutral", "confidence": 0.0-1.0, "rationale": "brief explanation"}

Rules:
- positive: bullish, growth, profit, success, upgrade
- negative: bearish, decline, loss, failure, downgrade
- neutral: mixed signals, uncertain, no clear direction
- confidence: how certain you are (1.0 = very certain, 0.5 = uncertain)
- rationale: one sentence explanation (max 50 words)

Respond with ONLY the JSON object, no other text.""",
                "user_template": "Analyze the financial sentiment of this text:\n\n{text}"
            }
        }

    def _detect_language(self, text: str) -> str:
        """Simple language detection (for future Japanese routing)"""
        # Check for Japanese characters
        if re.search(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]', text):
            return "ja"
        return "en"

    def _parse_response(self, content: str, model_id: str) -> SentimentResult:
        """Parse LLM response into standardized schema"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())

                # Normalize sentiment value
                sentiment = data.get("sentiment", "neutral").lower().strip()
                if sentiment not in ["positive", "negative", "neutral"]:
                    sentiment = "neutral"

                # Normalize confidence
                confidence = float(data.get("confidence", 0.5))
                confidence = max(0.0, min(1.0, confidence))

                return SentimentResult(
                    sentiment=sentiment,
                    confidence=confidence,
                    rationale=data.get("rationale", "")[:200],  # Truncate long rationale
                    model_id=model_id
                )

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
        except Exception as e:
            logger.error(f"Error parsing response: {e}")

        # Fallback: try to extract sentiment from text
        content_lower = content.lower()
        if "positive" in content_lower or "bullish" in content_lower:
            sentiment = "positive"
        elif "negative" in content_lower or "bearish" in content_lower:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return SentimentResult(
            sentiment=sentiment,
            confidence=0.3,  # Low confidence for fallback parsing
            rationale="Extracted from non-JSON response",
            model_id=model_id
        )

    async def analyze(
        self,
        input_data: SentimentInput,
        low_confidence_threshold: float = 0.5
    ) -> SentimentResult:
        """
        Analyze sentiment with standardized output.

        Args:
            input_data: Text to analyze
            low_confidence_threshold: Return neutral if below this threshold

        Returns:
            SentimentResult with standardized schema
        """
        # Detect language
        language = input_data.language or self._detect_language(input_data.text)

        # Get prompts
        prompts = self.prompts.get("sentiment", self._default_prompts()["sentiment"])

        # Build messages
        messages = [
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": prompts["user_template"].format(text=input_data.text)}
        ]

        # Route request through Policy Router
        payload = {
            "messages": messages,
            "task_type": "sentiment",
            "max_tokens": 256,
            "temperature": 0.1,  # Low temperature for consistency
            "language": language
        }

        async with httpx.AsyncClient(timeout=60) as client:
            try:
                response = await client.post(
                    f"{self.router_url}/v1/chat/completions",
                    json=payload
                )
                response.raise_for_status()
                result = response.json()

                # Extract model_id from routing info
                routing_info = result.get("_routing", {})
                model_id = routing_info.get("model", "unknown")

                # Parse response content
                content = result["choices"][0]["message"]["content"]
                sentiment_result = self._parse_response(content, model_id)

                # Apply low confidence safety rule
                if sentiment_result.confidence < low_confidence_threshold:
                    logger.info(
                        f"Low confidence ({sentiment_result.confidence:.2f}), "
                        f"returning neutral instead of {sentiment_result.sentiment}"
                    )
                    return SentimentResult(
                        sentiment="neutral",
                        confidence=sentiment_result.confidence,
                        rationale=f"Low confidence fallback. Original: {sentiment_result.rationale}",
                        model_id=model_id
                    )

                return sentiment_result

            except Exception as e:
                logger.error(f"Sentiment analysis failed: {e}")
                return SentimentResult(
                    sentiment="neutral",
                    confidence=0.0,
                    rationale=f"Analysis failed: {str(e)}",
                    model_id="error"
                )

    async def analyze_batch(
        self,
        texts: list[str],
        source: Optional[str] = None
    ) -> list[SentimentResult]:
        """Analyze multiple texts (for batch processing)"""
        results = []
        for text in texts:
            input_data = SentimentInput(text=text, source=source)
            result = await self.analyze(input_data)
            results.append(result)
        return results


# Convenience function
async def analyze_sentiment(text: str, language: Optional[str] = None) -> SentimentResult:
    """
    Quick sentiment analysis function.

    Example:
        result = await analyze_sentiment("Fed raised rates by 25bps")
        print(result.sentiment)  # "negative"
        print(result.confidence)  # 0.85
    """
    pipeline = SentimentPipeline()
    return await pipeline.analyze(SentimentInput(text=text, language=language))


# CLI for testing
if __name__ == "__main__":
    import asyncio
    import sys

    async def main():
        if len(sys.argv) < 2:
            print("Usage: python sentiment_pipeline.py <text>")
            print("Example: python sentiment_pipeline.py 'Fed raised rates'")
            return

        text = " ".join(sys.argv[1:])
        print(f"Analyzing: {text}")
        print("-" * 50)

        result = await analyze_sentiment(text)
        print(f"Sentiment: {result.sentiment}")
        print(f"Confidence: {result.confidence:.2%}")
        print(f"Rationale: {result.rationale}")
        print(f"Model: {result.model_id}")

    asyncio.run(main())
