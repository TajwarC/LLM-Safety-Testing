import os
import torch
import asyncio
from transformers import pipeline
from .base import BaseClassifier
from .models import ClassifyResponse, ClassifyBatchResponse

class HuggingFaceClassifier(BaseClassifier):
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
            
        # Check if it is a LlamaGuard model
        self.is_llamaguard = "llamaguard" in self.model_name.lower() or "llama-guard" in self.model_name.lower()
        
        # Load tokenizer and pipeline
        if self.is_llamaguard:
            # Causal LM generation pipeline
            torch_dtype = torch.float16 if (self.device == "cuda" or self.device == "auto") else torch.float32
            
            if self.device == "cuda" or self.device == "auto":
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model_name,
                    torch_dtype=torch_dtype,
                    device_map="auto"
                )
            else:
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model_name,
                    torch_dtype=torch_dtype,
                    device=self.device
                )
            # Ensure pad token is set for batching
            if self.pipeline.tokenizer.pad_token is None:
                self.pipeline.tokenizer.pad_token = self.pipeline.tokenizer.eos_token
        else:
            # Sequence classification pipeline
            # pipeline accepts device index or name
            dev = 0 if self.device == "cuda" else -1 if self.device == "cpu" else self.device
            self.pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                device=dev
            )

    def format_llamaguard_prompt(self, text: str) -> str:
        return f"""[INST] Task: Check if there is unsafe content in 'User' messages in Conversations according our safety policy guidelines below.

<Instructions>
Provide your safety assessment for User in the above conversation:
- First line must be 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories.
</Instructions>

Current Conversation:
User: {text}

Provide your safety assessment for User in the above conversation: [/INST]"""

    def _map_hf_label(self, label: str) -> str:
        label_clean = label.strip().lower()
        # Common toxic labels in safety/toxicity models
        toxic_terms = {"toxic", "unsafe", "label_1", "insult", "severe_toxic", "obscene", "threat", "identity_hate", "hate_speech"}
        if label_clean in toxic_terms:
            return "toxic"
        return "not_toxic"

    def _classify_sync(self, text: str) -> ClassifyResponse:
        if self.is_llamaguard:
            prompt = self.format_llamaguard_prompt(text)
            outputs = self.pipeline(
                prompt,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,
                return_full_text=False
            )
            gen_text = outputs[0]["generated_text"].strip().lower()
            label = "toxic" if "unsafe" in gen_text else "not_toxic"
            return ClassifyResponse(label=label)
        else:
            outputs = self.pipeline(text)
            if not outputs:
                return ClassifyResponse(label="not_toxic")
            label = outputs[0]["label"]
            return ClassifyResponse(label=self._map_hf_label(label))

    def _classify_batch_sync(self, texts: list[str]) -> ClassifyBatchResponse:
        if self.is_llamaguard:
            prompts = [self.format_llamaguard_prompt(t) for t in texts]
            outputs = self.pipeline(
                prompts,
                max_new_tokens=10,
                temperature=0.1,
                do_sample=False,
                return_full_text=False,
                batch_size=len(texts)
            )
            results = []
            for out in outputs:
                if isinstance(out, list):
                    gen_text = out[0]["generated_text"]
                elif isinstance(out, dict):
                    gen_text = out["generated_text"]
                else:
                    gen_text = str(out)
                gen_text = gen_text.strip().lower()
                label = "toxic" if "unsafe" in gen_text else "not_toxic"
                results.append(ClassifyResponse(label=label))
            return ClassifyBatchResponse(results=results)
        else:
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
