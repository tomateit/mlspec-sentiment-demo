from ..AbstractVectorizer import AbstractVectorizer
import os
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import html
import dill
from typing import List


class DummyPreprocessor(TransformerMixin):
    def __init__(self):
        pass
    
    def fit_transform(self, data, y=None):
        return data
    
    def transform(self, data, y=None):
        return data
    
    def fit(self, data, y=None):
        return data

class Vectorizer(AbstractVectorizer):
    __instance = None
    __model = None

    def __new__(cls):
        if cls.__instance is None:
            with open(os.path.join(os.path.dirname(__file__), "Vectorizer.pkl"), "rb") as fin:
                print("Загружаю векторайзер для ру")
                cls.__model = dill.load(fin)
            cls.__instance = super(Vectorizer, cls).__new__(cls)
        return cls.__instance

    def _preprocess_input(self, incoming_text: str) -> List[str]:
        """Preprocess input text"""
        return RoughPreprocessor().fit_transform([incoming_text])

    def _vectorize_input(self, preprocessed_text: List[str]):
        """Vectorize text accordingly to expected classifier input"""
        return self.__model.transform(preprocessed_text)

    
            


class RoughPreprocessor(TransformerMixin):
    def __init__(self):
        pass
    
    def fit_transform(self, data: List[str], y=None):
        assert type(data) is list, f"Unsupported type {type(data)}"
        return list(map(self.normalize_text_re, map(self.normalize_text, data)))
    
    def normalize_text(self, text: str):
        assert type(text) is str, f"Unsupported type {type(text)}"
        _t = html.unescape(text)
        _t = _t.replace("\n", " ")
        _t = _t.replace("\r", " ")
        _t = _t.replace("\t", " ")
        _t = _t.replace('"', " ")
        _t = _t.lower()
        return _t.strip()
    
    def normalize_text_re(self, text: str):
        _t = text
        _t = re.sub(r"[\s.,\-\+><;:!?()]", " ", _t)
        _t = re.sub(r"\s+", " ", _t)
        return _t.strip()
    
    def fit(self, data: List[str], y=None):
        return self.fit_transform(data)
    
    def transform(self, data: List[str], y=None):
        return self.fit_transform(data) 


