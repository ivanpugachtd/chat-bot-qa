from typing import List, Self
from numpy.typing import NDArray
import numpy as np
from sentence_transformers import SentenceTransformer
from src.data_manager.splitters import ParagraphSplitter
from src.data_manager.file_loader import BaseLoader
from src.data_manager.text_preprocessor import TextPreprocessorSimple
from scipy.sparse import spmatrix
import faiss
from httpx import AsyncClient, Timeout, HTTPStatusError



class OpenAiService:
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._prompt = """You are a helpful assistant. 
                          Answer as humanly as possible.
                          You should answer the question based on the text provided with question
                        """
        self._completion_url = "https://api.openai.com/v1/chat/completions"
        self._default_model = "gpt-4o-mini"
        self._max_tokens = 100
        self._httpx_timeout = Timeout(10)

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

    async def get_response(self, question: str, source_data: str) -> str:
        messages = [
            {"role": "system", "content": self._prompt},
            {"role": "user", "content": f"{question}\n\nYour source data for providing answers {source_data}"},
        ]
        try:
            async with AsyncClient(timeout=self._httpx_timeout) as client:
                response = await client.post(
                    self._completion_url,
                    headers=self.get_headers(),
                    json={
                        "model": self._default_model,
                        "messages": messages,
                        "max_tokens": 100,
                    },
                )
                response_json = response.json()
                return response_json["choices"][0]["message"]["content"]
        except HTTPStatusError:
            return "Error occurred while getting response"
            


class ChatBotServiceSentenceTransformer:
    def __init__(self, openai_service: OpenAiService, data_loader: BaseLoader) -> None:
        self._paragraphs: List[str] = []
        self.processed_source: spmatrix | None = None
        self._data_loader = data_loader
        self._model = None
        self._openai_service = openai_service

    def process(self, text_preprocessor: TextPreprocessorSimple) -> Self:
        book_text = self._data_loader.get_data()
        if not book_text:
            raise ValueError("Book text is empty")

        self._paragraphs = ParagraphSplitter(book_text).get_splitted_data()
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        self._paragraphs = list(
            (
                paragraph
                for paragraph in text_preprocessor.preprocess_text(self._paragraphs)
            )
        )

        paragraph_embeddings = self._model.encode(
            self._paragraphs,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        paragraph_embeddings: NDArray[np.float32] = np.array(
            paragraph_embeddings
        ).astype("float32")
        faiss.normalize_L2(paragraph_embeddings)
        dimension = paragraph_embeddings.shape[1]
        self._index = faiss.IndexFlatL2(dimension)
        self._index.add(paragraph_embeddings)  # type: ignore

        return self

    async def get_answer(self, question: str, top_idx: int = 3) -> str:
        if self._model is None:
            raise ValueError("Model is not initialized")

        question_embedding: NDArray[np.float32] = self._model.encode(
            [question], convert_to_numpy=True
        ).astype("float32")
        faiss.normalize_L2(question_embedding)

        _, indices = self._index.search(question_embedding, top_idx)  # type: ignore
        if len(indices) == 0:
            raise ValueError("No similarities found")
        
        similar_paragraphs = [self._paragraphs[idx] for idx in indices[0]]
        
        return await self.get_response_from_model(question, "\n\n".join(similar_paragraphs[:top_idx]))

    async def get_response_from_model(self, question: str, source_data: str) -> str:
        return await self._openai_service.get_response(question, source_data)



        
