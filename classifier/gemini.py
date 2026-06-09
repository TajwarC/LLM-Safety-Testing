import os
import asyncio
from google import genai
from google.genai import types
from .base import BaseClassifier
from .models import ClassifyResponse, ClassifyBatchResponse
from .prompts import SYSTEM_PROMPT

class GeminiClassifier(BaseClassifier):
    def __init__(self, client: genai.Client = None, api_key: str = None, model_name: str = "gemini-3.1-flash-lite"):
        if client is not None:
            self.client = client
        else:
            key = api_key or os.environ.get("GEMINI_API_KEY")
            self.client = genai.Client(api_key=key)
        self.model_name = model_name

    async def classify(self, text: str) -> ClassifyResponse:
        resp = await self.client.aio.models.generate_content(
            model=self.model_name,
            contents=text,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=ClassifyResponse,
            )
        )
        return resp.parsed

    async def classify_batch(self, texts: list[str]) -> ClassifyBatchResponse:
        results = await asyncio.gather(*[self.classify(t) for t in texts])
        return ClassifyBatchResponse(results=list(results))
