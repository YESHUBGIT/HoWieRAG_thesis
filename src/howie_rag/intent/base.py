from abc import ABC, abstractmethod

from howie_rag.core.schemas import IntentResult


class BaseIntentClassifier(ABC):
    @abstractmethod
    def classify(self, question: str) -> IntentResult:
        raise NotImplementedError
