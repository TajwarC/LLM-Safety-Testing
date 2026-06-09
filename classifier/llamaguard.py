import os
import torch
import asyncio
from transformers import pipeline
from .base import BaseClassifier
from .models import ClassifyResponse, ClassifyBatchResponse

class LlamaGuardClassifier(BaseClassifier):
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

    def _classify_sync(self, text: str) -> ClassifyResponse:
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

    def _classify_batch_sync(self, texts: list[str]) -> ClassifyBatchResponse:
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

    async def classify(self, text: str) -> ClassifyResponse:
        # Run in thread executor to keep FastAPI async loop responsive
        return await asyncio.to_thread(self._classify_sync, text)

    async def classify_batch(self, texts: list[str]) -> ClassifyBatchResponse:
        # Run in thread executor to keep FastAPI async loop responsive
        return await asyncio.to_thread(self._classify_batch_sync, texts)
