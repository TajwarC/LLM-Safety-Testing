import os
import asyncio
from fastapi import FastAPI
from openai import AsyncOpenAI
from .models import ClassifyRequest, ClassifyResponse, ClassifyBatchRequest, ClassifyBatchResponse
from .prompts import SYSTEM_PROMPT

app = FastAPI()
client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])


@app.get("/health")
def health():
    return {"status": "ok"}


async def _classify(text: str) -> ClassifyResponse:
    resp = await client.chat.completions.create(
        model="gpt-5.4-nano",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
    )
    return ClassifyResponse.model_validate_json(resp.choices[0].message.content)


@app.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest):
    return await _classify(req.text)


@app.post("/classify_batch", response_model=ClassifyBatchResponse)
async def classify_batch(req: ClassifyBatchRequest):
    results = await asyncio.gather(*[_classify(t) for t in req.texts])
    return ClassifyBatchResponse(results=list(results))
