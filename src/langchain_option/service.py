

import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from typing import Self


class ChatBotLangchain:
    def __init__(self, book_path: str, openai_key: str) -> None:
        self.book_path = book_path
        self.openai_key = openai_key
        self._chunk_size = 1000
        self._chunk_overlap = 200
        self.book = None
        self._local_vector_store = "faiss_local_index"
        self._separators =["\n\n", "\n", " ", ""]
        self._default_model = "gpt-3.5-turbo"
        
        
    def process(self) -> Self:
        if self.openai_key is None or self.book_path is None:
            raise ValueError(
                "Incorrect environment variables, make sure to set OPENAI_KEY and BOOK_PATH"
            )
        
        if not os.path.exists(self.book_path):
            raise ValueError("Book path does not exist")
        
        loader = TextLoader(self.book_path)
        book = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            separators=self._separators
        )
        self.book = text_splitter.split_documents(book)
        embeddings = OpenAIEmbeddings()
        if not os.path.exists(self._local_vector_store):
            vector_store = FAISS.from_documents(self.book, embeddings)
            vector_store.save_local(self._local_vector_store)
            
        _vector_store = FAISS.load_local(self._local_vector_store, embeddings, allow_dangerous_deserialization=True)
        llm = ChatOpenAI(model_name=self._default_model, temperature=0)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=_vector_store.as_retriever()
        )
        return self
    
    def get_answer(self, question: str) -> str:
        response = self.qa_chain.run(question)
        return response