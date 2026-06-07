import os
import asyncio
from fastapi import FastAPI
from google import genai
from google.genai import types
from .models import ClassifyRequest, ClassifyResponse, ClassifyBatchRequest, ClassifyBatchResponse
from .prompts import SYSTEM_PROMPT

# Load environment variables from .env if it exists at the root of the project
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(root_dir, ".env")
if os.path.exists(env_path):
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip().strip('"').strip("'")

app = FastAPI()
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


@app.get("/health")
def health():
    return {"status": "ok"}


async def _classify(text: str) -> ClassifyResponse:
    resp = await client.aio.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=text,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=ClassifyResponse,
        )
    )
    return resp.parsed


@app.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest):
    return await _classify(req.text)


@app.post("/classify_batch", response_model=ClassifyBatchResponse)
async def classify_batch(req: ClassifyBatchRequest):
    results = await asyncio.gather(*[_classify(t) for t in req.texts])
    return ClassifyBatchResponse(results=list(results))
