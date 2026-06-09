from abc import ABC, abstractmethod
from .models import ClassifyResponse, ClassifyBatchResponse

class BaseClassifier(ABC):
    @abstractmethod
    async def classify(self, text: str) -> ClassifyResponse:
        """
        Classifies a single text string as toxic or not_toxic.
        """
        pass

    @abstractmethod
    async def classify_batch(self, texts: list[str]) -> ClassifyBatchResponse:
        """
        Classifies a batch of text strings as toxic or not_toxic.
        """
        pass
