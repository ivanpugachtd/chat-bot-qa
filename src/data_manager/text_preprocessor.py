import string
from typing import Generator, List, Self, Type
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import spmatrix


class SentencesProcessor:
    def __init__(self, text: str, language: str = "english") -> None:
        self._text = text
        self._language = language
        self._sentences: List[str] | None = None

    def _init(self) -> Self:
        self._sentences = sent_tokenize(self._text, self._language)
        print(f"Sentences type: {type(self._sentences)}")
        return self

    def __getitem__(self, index: int) -> str:
        if self._sentences is None:
            self._init()
        if not self._sentences:
            raise ValueError("No sentences found")
        if not 0 <= index < len(self._sentences):
            raise IndexError("Index out of range")

        return self._sentences[index]

    def get_sentences(self) -> List[str]:
        if self._sentences is None:
            self._init()

        if not self._sentences:
            raise ValueError("No sentences found")

        return self._sentences


class TextPreprocessor:
    def __init__(self, vectorizer: TfidfVectorizer) -> None:
        self.stop_words = set(stopwords.words("english"))
        self.punctuation = set(string.punctuation)
        self.vectorizer = vectorizer

    def _preprocess_sentence(self, sentence: str) -> str:
        sentence = sentence.lower()
        words = word_tokenize(sentence)
        words = [
            word
            for word in words
            if word not in self.stop_words and word not in self.punctuation
        ]
        return " ".join(words)

    def preprocess_text(self, sentences: List[str]) -> Generator[str, None, None]:
        preprocessed_sentences = (
            self._preprocess_sentence(sentence) for sentence in sentences
        )
        return preprocessed_sentences

    def vectorize(self, sentences: List[str]) -> spmatrix:
        return self.vectorizer.fit_transform(self.preprocess_text(sentences))

    def transform(self, question: str) -> spmatrix:
        return self.vectorizer.transform([self._preprocess_sentence(question)])
