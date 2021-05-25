from ..AbstractVectorizer import AbstractVectorizer
from typing import List
from transformers import AutoTokenizer

class Vectorizer(AbstractVectorizer):
    __instance = None
    __model = None
    def __new__(cls):
        if cls.__instance is None:
            cls.__model = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
            cls.__instance = super(Vectorizer, cls).__new__(cls)
        return cls.__instance

    def _preprocess_input(self, incoming_text: str) -> List[str]:
        """Preprocess input text"""
        return [incoming_text]

    def _vectorize_input(self, preprocessed_text: List[str]):
        """Vectorize text accordingly to expected classifier input"""
        return self.__model(preprocessed_text, truncation=True, padding=True, return_tensors="pt")



