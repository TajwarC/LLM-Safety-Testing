import os
import asyncio
from openai import AsyncOpenAI
from .base import BaseClassifier
from .models import ClassifyResponse, ClassifyBatchResponse
from .prompts import SYSTEM_PROMPT

class OpenAIClassifier(BaseClassifier):
    def __init__(self, api_key: str = None, model_name: str = "gpt-4o-mini"):
        key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=key)
        self.model_name = model_name

    async def classify(self, text: str) -> ClassifyResponse:
        resp = await self.client.beta.chat.completions.parse(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text}
            ],
            response_format=ClassifyResponse,
        )
        parsed = resp.choices[0].message.parsed
        if parsed is None:
            # Fallback if parsing fails (highly unlikely with structured outputs)
            return ClassifyResponse(label="not_toxic")
        return parsed

    async def classify_batch(self, texts: list[str]) -> ClassifyBatchResponse:
        results = await asyncio.gather(*[self.classify(t) for t in texts])
        return ClassifyBatchResponse(results=list(results))
