from ..AbstractClassifier import AbstractClassifier
from .Vectorizer import Vectorizer
import os
import pickle
import logging

class Classifier(AbstractClassifier):
    def __init__(self):
        with open(os.path.join(os.path.dirname(__file__), "Classifier.pkl"), "rb") as mfile:
            self.model = pickle.load(mfile)
        self.classnames = {0: "негативный", 1: "позитивный"}

    def _preprocess_input(self, incoming_text: str):
        """First we preprocess input text"""
        return Vectorizer().vectorize_input(incoming_text)

    def _get_human_readable_interpretation(self, classname: int, score: float) -> str:
        _prob = round(score, 2)
        _classname = self.classnames.get(classname, "[Неподдерживаемый класс]")
        if score == -1:
            return " " + _classname
        if score < 0.55:
            return f" с небольшой вероятностью ({_prob}) {_classname}"
        if score < 0.7:
            return f" возможно ({_prob}) {_classname}"
        if score > 0.95:
            return f" наиболее вероятно ({_prob}) {_classname}"
        else:
            raise Exception("Impossible state")

    def get_prediction(self, incoming_text: str) -> str:
        try:
            return self._get_prediction(incoming_text)
        except Exception as e:
            logging.exception("Prediction error:")
            return "Извините, возникла ошибка."