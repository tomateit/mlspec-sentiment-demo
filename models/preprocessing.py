from sklearn.base import TransformerMixin, BaseEstimator
import re
import html

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

def preprocessor_ru(text: str):
    