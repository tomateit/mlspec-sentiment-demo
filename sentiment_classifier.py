import pickle
import dill

class SentimentClassifierEN(object):
    def __init__(self):
        with open("./SentimentModelEN.pkl", "rb") as mfile:
            self.model = pickle.load(mfile)
            # self.vectorizer = pickle.load("./BigramUnprocessedVectorizer.pkl")
        self.classes_dict = {0: "negative", 1: "positive", -1: "prediction error"}

    @staticmethod
    def get_probability_words(probability):
        _prob = round(probability, 2)
        if probability < 0.55:
            return f"neutral or uncertain ({_prob})"
        if probability < 0.7:
            return f"probably ({_prob})"
        if probability > 0.95:
            return f"certain ({_prob})"
        else:
            return ""

    def predict_text(self, text):
        try:
            sent_class = self.model.predict([text])[0]
            probability = self.model.predict_proba([text])[0].max()
            return (sent_class, probability)
        except:
            print("prediction error")
            return -1, 0.8


    def get_prediction_message(self, text):
        prediction = self.predict_text(text)
        class_prediction = prediction[0]
        prediction_probability = prediction[1]
        return self.get_probability_words(prediction_probability) + " " + self.classes_dict[class_prediction]

class SentimentClassifierRU(object):
    def __init__(self):
        with open("./SentimentModelRU.pkl", "rb") as mfile:
            self.model = dill.load(mfile)
        self.classes_dict = {0: "негативный", 1: "позитивный", -1: "ошибка"}

    # @staticmethod
    #def get_probability_words(probability):
        #_prob = round(probability, 2)
        #if probability < 0.55:
            #return f"предположительно ({_prob})"
        #if probability < 0.7:
            #return f"вероятно ({_prob})"
        #if probability > 0.95:
            #return f"определённо ({_prob})"
        #else:
            #return ""

    def predict_text(self, text):
        try:
            sent_class = self.model.predict([text])[0]
            #probability = self.model.predict_proba([text])[0].max()
            return sent_class
            #return (sent_class, probability)
        except:
            print("prediction error")
            #return -1, 0.8
            return -1

    def get_prediction_message(self, text):
        prediction = self.predict_text(text)
        #class_prediction = prediction[0]
        #prediction_probability = prediction[1]
        return f"  {self.classes_dict[prediction]}"

        
