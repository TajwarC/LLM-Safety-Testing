import os
import torch
import asyncio
from transformers import pipeline
from .base import BaseClassifier
from .models import ClassifyResponse, ClassifyBatchResponse

class ToxicBertClassifier(BaseClassifier):
    def __init__(self, model_name_or_path: str, device: str = None):
        self.model_name = model_name_or_path
        
        # Determine device
        if device is not None:
            # If device is an integer string, convert to int
            if isinstance(device, str) and device.isdigit():
                self.device = int(device)
            else:
                self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        # Sequence classification pipeline
        # pipeline accepts device index or name
        dev = 0 if self.device == "cuda" else -1 if self.device == "cpu" else self.device
        self.pipeline = pipeline(
            "text-classification",
            model=self.model_name,
            device=dev
        )

    def _map_hf_label(self, label: str) -> str:
        label_clean = label.strip().lower()
        # Common toxic labels in safety/toxicity models
        toxic_terms = {"toxic", "unsafe", "label_1", "insult", "severe_toxic", "obscene", "threat", "identity_hate", "hate_speech"}
        if label_clean in toxic_terms:
            return "toxic"
        return "not_toxic"

    def _classify_sync(self, text: str) -> ClassifyResponse:
        outputs = self.pipeline(text)
        if not outputs:
            return ClassifyResponse(label="not_toxic")
        label = outputs[0]["label"]
        return ClassifyResponse(label=self._map_hf_label(label))

    def _classify_batch_sync(self, texts: list[str]) -> ClassifyBatchResponse:
        outputs = self.pipeline(texts, batch_size=len(texts))
        results = []
        for out in outputs:
            if isinstance(out, list):
                label = out[0]["label"] if out else "LABEL_0"
            elif isinstance(out, dict):
                label = out["label"]
            else:
                label = "LABEL_0"
            results.append(ClassifyResponse(label=self._map_hf_label(label)))
        return ClassifyBatchResponse(results=results)

    async def classify(self, text: str) -> ClassifyResponse:
        # Run in thread executor to keep FastAPI async loop responsive
        return await asyncio.to_thread(self._classify_sync, text)

    async def classify_batch(self, texts: list[str]) -> ClassifyBatchResponse:
        # Run in thread executor to keep FastAPI async loop responsive
        return await asyncio.to_thread(self._classify_batch_sync, texts)
