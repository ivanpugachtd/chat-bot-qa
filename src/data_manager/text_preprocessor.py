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
        self._sentences: List[str] = []
        
    def init(self) -> Self:
        self._sentences = sent_tokenize(self._text, self._language)
        print(f"Sentences type: {type(self._sentences)}")
        return self
        
    def __getitem__(self, index: int) -> str:
        return self._sentences[index]
        

class TextPreprocessor:
    def __init__(self) -> None:
        self.stop_words = set(stopwords.words("english"))
        self.punctuation = set(string.punctuation)

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
    
    def vectorize(self, sentences: List[str], vectorizer: TfidfVectorizer) -> spmatrix:
        return vectorizer.fit_transform(self.preprocess_text(sentences))
