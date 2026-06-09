import os
import asyncio
from fastapi import FastAPI
from google import genai

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

# To preserve the API client reference for existing unit tests
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

from .models import ClassifyRequest, ClassifyResponse, ClassifyBatchRequest, ClassifyBatchResponse
from .base import BaseClassifier
from .gemini import GeminiClassifier
from .openai import OpenAIClassifier
from .huggingface import HuggingFaceClassifier

app = FastAPI()
classifier_instance = None

def get_classifier() -> BaseClassifier:
    model_type = os.environ.get("CLASSIFIER_TYPE")
    model_name = os.environ.get("CLASSIFIER_MODEL")
    device = os.environ.get("CLASSIFIER_DEVICE")

    # If neither is set, default to gemini
    if not model_type and not model_name:
        return GeminiClassifier(client=client)

    # Determine type if not explicitly set
    if not model_type:
        name_lower = model_name.lower()
        if name_lower.startswith("gemini"):
            model_type = "gemini"
        elif name_lower.startswith("gpt-") or name_lower == "openai":
            model_type = "openai"
        else:
            model_type = "huggingface"

    if model_type == "gemini":
        model_name = model_name or "gemini-3.1-flash-lite"
        return GeminiClassifier(client=client, model_name=model_name)
    elif model_type == "openai":
        model_name = model_name or "gpt-4o-mini"
        return OpenAIClassifier(model_name=model_name)
    elif model_type == "huggingface":
        if not model_name:
            raise ValueError("Hugging Face model name/path must be specified via CLASSIFIER_MODEL")
        return HuggingFaceClassifier(model_name_or_path=model_name, device=device)
    else:
        raise ValueError(f"Unknown classifier type: {model_type}")


def get_classifier_instance() -> BaseClassifier:
    global classifier_instance
    if classifier_instance is None:
        classifier_instance = get_classifier()
    return classifier_instance


@app.on_event("startup")
def startup_event():
    get_classifier_instance()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest):
    clf = get_classifier_instance()
    return await clf.classify(req.text)


@app.post("/classify_batch", response_model=ClassifyBatchResponse)
async def classify_batch(req: ClassifyBatchRequest):
    clf = get_classifier_instance()
    return await clf.classify_batch(req.texts)


def start():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Start the classifier FastAPI server.")
    parser.add_argument("--model", type=str, default=None, help="Model type or path (e.g. gemini, openai, meta-llama/LlamaGuard-7b, unitary/toxic-bert)")
    parser.add_argument("--type", type=str, default=None, choices=["gemini", "openai", "huggingface"], help="Explicitly set classifier backend type")
    parser.add_argument("--device", type=str, default=None, help="Device for Hugging Face inference (e.g. cuda, cpu, 0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the FastAPI server on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind the server to")
    
    args = parser.parse_args()

    # Pass the CLI arguments to environment variables so FastAPI can read them on startup
    if args.model:
        os.environ["CLASSIFIER_MODEL"] = args.model
    if args.type:
        os.environ["CLASSIFIER_TYPE"] = args.type
    if args.device:
        os.environ["CLASSIFIER_DEVICE"] = args.device

    print(f"Starting classifier server with model: {args.model or 'gemini (default)'} on port {args.port}...")
    uvicorn.run("classifier.main:app", host=args.host, port=args.port, reload=False)
