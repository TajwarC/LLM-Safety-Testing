from .base import BaseClassifier

class HuggingFaceClassifier(BaseClassifier):
    def __new__(cls, model_name_or_path: str, device: str = None):
        model_name_lower = model_name_or_path.lower()
        if "llamaguard" in model_name_lower or "llama-guard" in model_name_lower:
            from .llamaguard import LlamaGuardClassifier
            return LlamaGuardClassifier(model_name_or_path=model_name_or_path, device=device)
        else:
            from .toxic_bert import ToxicBertClassifier
            return ToxicBertClassifier(model_name_or_path=model_name_or_path, device=device)

    async def classify(self, text: str):
        # This is never called because __new__ returns an instance of LlamaGuardClassifier or ToxicBertClassifier,
        # but is declared to satisfy abstract method checks if any.
        pass

    async def classify_batch(self, texts: list[str]):
        # This is never called because __new__ returns an instance of LlamaGuardClassifier or ToxicBertClassifier,
        # but is declared to satisfy abstract method checks if any.
        pass
