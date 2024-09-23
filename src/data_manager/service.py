from typing import List, Self
from src.data_manager.splitters import SentencesSplitter
from src.data_manager.file_loader import BaseLoader
from src.data_manager.text_preprocessor import TextPreprocessorSimple
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import spmatrix


class ChatBotServiceOption1:
    def __init__(
        self, qa_processor: TextPreprocessorSimple, data_loader: BaseLoader
    ) -> None:
        self.qa_processor = qa_processor
        self._sentences: List[str] = []
        self.processed_source: spmatrix | None = None
        self._data_loader = data_loader

    def process(self) -> Self:
        book_text = self._data_loader.get_data()
        if not book_text:
            raise ValueError("Book text is empty")

        sentences_preprocessor = SentencesSplitter(book_text)
        self._sentences = sentences_preprocessor.get_splitted_data()
        self.processed_source = self.qa_processor.preprocess_source(self._sentences)

        return self

    def get_answer(self, question: str, top_idx: int = 1) -> str:
        question_vector = self.qa_processor.preprocess_question(question)
        similarities = cosine_similarity(question_vector, self.processed_source)
        if len(similarities) == 0:
            raise ValueError("No similarities found")

        top_indices = similarities[0].argsort()[-top_idx:][::-1]
        answers = [self._sentences[idx] for idx in top_indices]
        if not answers:
            return "No answers found"
        return answers[0]
