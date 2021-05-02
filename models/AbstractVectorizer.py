from abc import abstractmethod, ABCMeta

class AbstractVectorizer(metaclass=ABCMeta):
    __instance = None
    def __call__(self):
        if self.__instance is None:
            self.__instance = super(AbstractVectorizer, self).__call__()
            print("Vectorizer instanciated")
        print("Vectorizer created")
        return self.__instance

    @abstractmethod
    def _preprocess_input(self, incoming_text: str):
        """Preprocess input text"""
        raise NotImplemented

    @abstractmethod
    def _vectorize_input(self, text: str):
        """Vectorize text accordingly to expected classifier input"""
        raise NotImplemented

    def vectorize_input(self, incoming_text: str):
        return self._preprocess_input(self._vectorize_input)
