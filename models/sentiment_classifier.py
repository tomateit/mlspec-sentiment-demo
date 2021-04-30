import pickle
import dill
import logging
from abc import abstractmethod, ABCMeta
from .preprocessing import preprocess_ru, preprocess_en
import os


class IPredictor(metaclass=ABCMeta):
    @abstractmethod
    def _preprocess_input(self, incoming_text: str):
        """First we preprocess input text"""
        raise NotImplemented

    def _predict_sentiment(self, preprocessed_input) -> (int, float):
        """
        Second we get predict class and probability of belonging the text to positive (1) class.
        Model will be called predict_proba attribute, and if it has not got one, then score will be assigned -1 
        """
        classname = self.model.predict(preprocessed_input)[0]
        if hasattr(self.model, "predict_proba"):
            score = self.model.predict_proba(preprocessed_input)[0]
        else:
            score = -1.
        print(f"Predicted sentimen class {classname} with certainity {score}")
        return (classname, score)
        

    @abstractmethod
    def _get_human_readable_interpretation(self, classname: int, score: float) -> str:
        """Sentiment score can be converted to a word"""
        raise NotImplemented


    def _get_prediction(self, incoming_text: str) -> str:
        preprocessed_input = self._preprocess_input(incoming_text)
        classname, score = self._predict_sentiment(preprocessed_input)
        message = self._get_human_readable_interpretation(classname, score)
        return message

    @abstractmethod
    def get_prediction(self, incoming_text: str) -> str:
        """Class shall implement exception-aware variant of _get_prediction"""
        raise NotImplemented


class SentimentClassifierEN(IPredictor):
    def __init__(self):
        with open(os.path.join(os.path.dirname(__file__), "SentimentModelEN.pkl"), "rb") as mfile:
            self.model = pickle.load(mfile)
        self.classnames = {0: "negative", 1: "positive"}

    def _preprocess_input(self, incoming_text: str):
        """First we preprocess input text"""
        return preprocess_en(incoming_text)


    def _get_human_readable_interpretation(self, classname: int, score: float) -> str:
        _prob = round(score, 2)
        _classname = self.classnames.get(classname, "Impossible class")
        if score == -1:
            return " " + _classname
        if score < 0.55:
            return f" neutral or uncertain ({_prob}) {_classname}"
        if score < 0.7:
            return f" probably ({_prob}) {_classname}"
        if score > 0.95:
            return f" certain ({_prob}) {_classname}"
        else:
            raise Exception("Impossible state")



    def get_prediction(self, incoming_text: str) -> str:
        try:
            return self._get_prediction(incoming_text)
        except Exception as e:
            logging.exception("Prediction error:")
            return "An exception occured. Please, refer to logs."


class SentimentClassifierRU(IPredictor):
    def __init__(self):
        with open(os.path.join(os.path.dirname(__file__), "SentimentModelRU.pkl"), "rb") as mfile:
            self.model = pickle.load(mfile)
        self.classnames = {0: "негативный", 1: "позитивный"}

    def _preprocess_input(self, incoming_text: str):
        """First we preprocess input text"""
        return preprocess_ru(incoming_text)

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

        
