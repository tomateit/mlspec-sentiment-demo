from abc import abstractmethod, ABCMeta

class AbstractVectorizer(metaclass=ABCMeta):

    # @property
    # @classmethod
    # def __instance(cls):
    #     return NotImplementedError
    # def __new__(cls):
    #     if self.__instance is None:
    #         self.__instance = super(AbstractVectorizer, self).__call__()
    #         print("Vectorizer instanciated")
    #     print("Vectorizer created")
    #     return self.__instance

    @abstractmethod
    def _preprocess_input(self, incoming_text: str):
        """Preprocess input text"""
        raise NotImplemented

    @abstractmethod
    def _vectorize_input(self, text: str):
        """Vectorize text accordingly to expected classifier input"""
        raise NotImplemented

    def vectorize_input(self, incoming_text: str):
        return self._vectorize_input(self._preprocess_input(incoming_text))