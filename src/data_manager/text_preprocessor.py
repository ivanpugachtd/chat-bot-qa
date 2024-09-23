import string
from typing import Generator, List, Protocol
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import spmatrix

class BaseProcessor(Protocol):
    def preprocess_source(self, ):
        raise NotImplementedError("Method preprocess_source not implemented")
    
    def preprocess_question(self):
        raise NotImplementedError("Method preprocess_question not implemented")
    

class TextPreprocessorSimple(BaseProcessor):
    def __init__(self) -> None:
        self.stop_words = set(stopwords.words("english"))
        self.punctuation = set(string.punctuation)
        self.vectorizer = TfidfVectorizer()

    def _preprocess_chunk(self, chunk: str) -> str:
        chunk = chunk.lower()
        words = word_tokenize(chunk)
        words = [
            word
            for word in words
            if word not in self.stop_words and word not in self.punctuation
        ]
        return " ".join(words)

    def _preprocess_text(self, chunks: List[str]) -> Generator[str, None, None]:
        preprocessed_chunks = (
            self._preprocess_chunk(sentence) for sentence in chunks
        )
        return preprocessed_chunks

    def preprocess_source(self, sentences: List[str]) -> spmatrix:
        return self.vectorizer.fit_transform(self._preprocess_text(sentences))

    def preprocess_question(self, question: str) -> spmatrix:
        return self.vectorizer.transform([self._preprocess_chunk(question)])
