from abc import abstractmethod, ABCMeta

class AbstractClassifier(metaclass=ABCMeta):
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
            score = self.model.predict_proba(preprocessed_input)[0][classname] # default format is np([[prob0, prob1]])
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