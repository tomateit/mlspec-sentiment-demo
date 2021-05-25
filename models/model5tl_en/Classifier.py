from ..AbstractClassifier import AbstractClassifier
from .Vectorizer import Vectorizer
import logging
import torch
from transformers import AutoModelForSequenceClassification

class Classifier(AbstractClassifier):
    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.classnames = {0: "negative", 1: "positive"}

    def _preprocess_input(self, incoming_text: str):
        """First we preprocess input text"""
        return Vectorizer().vectorize_input(incoming_text)

    def _predict_sentiment(self, preprocessed_input) -> (int, float):
        logits = self.model(**preprocessed_input).logits
        classname = torch.argmax(logits, axis=1)[0].item()
        score = torch.sigmoid(logits)[0][classname].item()
        return (classname, score)
    
    def _get_prediction(self, incoming_text: str) -> str:
        preprocessed_input = self._preprocess_input(incoming_text)
        classname, score = self._predict_sentiment(preprocessed_input)
        message = self._get_human_readable_interpretation(classname, score)
        return message
    
    def _get_human_readable_interpretation(self, classname: int, score: float) -> str:
        _prob = round(score, 2)
        _classname = self.classnames.get(classname, "Impossible class")
        if score == -1:
            return " " + _classname
        if score < 0.55:
            return f" neutral or uncertain ({_prob}) {_classname}"
        if score < 0.7:
            return f" probably ({_prob}) {_classname}"
        if score < 1:
            return f" certain ({_prob}) {_classname}"
        else:
            raise Exception("Impossible state")



    def get_prediction(self, incoming_text: str) -> str:
        try:
            return self._get_prediction(incoming_text)
        except Exception as e:
            logging.exception("Prediction error:")
            return "An exception occured. Please, refer to logs."