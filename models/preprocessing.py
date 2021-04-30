from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import html
import dill

with open("./VectorizerRU.pkl", "rb") as fin:
    vectorizer_ru = dill.load(fin)

# ! WRONG FILENAME
with open("./VectorizerRU.pkl", "rb") as fin:
    vectorizer_en = dill.load(fin)


class DummyPreprocessor(TransformerMixin):
    def __init__(self):
        pass
    
    def fit_transform(self, data, y=None):
        return data
    
    def transform(self, data, y=None):
        return data
    
    def fit(self, data, y=None):
        return data


class RoughPreprocessor(TransformerMixin):
    def __init__(self):
        pass
    
    def fit_transform(self, data, y=None):
        return list(map(self.normalize_text_re, map(self.normalize_text, data)))
    
    def normalize_text(self, text):
        _t = html.unescape(text)
        _t = _t.replace("\n", " ")
        _t = _t.replace("\r", " ")
        _t = _t.replace("\t", " ")
        _t = _t.replace('"', " ")
        _t = _t.lower()
        return _t.strip()
    
    def normalize_text_re(self, text):
        _t = text
        _t = re.sub(r"[\s.,\-\+><;:!?()]", " ", _t)
        _t = re.sub(r"\s+", " ", _t)
        return _t.strip()
    
    def fit(self, data, y=None):
        return self.fit_transform(data)
    
    def transform(self, data, y=None):
        return self.fit_transform(data) 

def preprocess_ru(text: str):
    preprocessed = RoughPreprocessor().fit_transform([text])
    # print(preprocessed)
    vectorized = vectorizer_ru.transform(preprocessed)
    # print(vectorized)
    return vectorized

def preprocess_en(text: str):
    preprocessed = RoughPreprocessor().fit_transform([text])
    vectorized = vectorizer_ru.transform(preprocessed)
    return vectorized