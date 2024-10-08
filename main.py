from fastapi import FastAPI
from src.langchain_chatgpt.service import ChatBotLangchain
from src.chatgpt_with_faiss.service import ChatBotServiceSentenceTransformer, OpenAiService
from src.data_manager.file_loader import FileDataLoader
import logging
from dotenv import load_dotenv
from src.data_manager.models import Answer, QuestionReqeust
from src.data_manager.text_preprocessor import TextPreprocessorSimple
from src.data_manager.service import ChatBotServiceSimilarity
import os

load_dotenv()
book_path = os.environ.get("BOOK_PATH", None)
open_api_key = os.environ.get("OPENAI_API_KEY", None)
PORT= int(os.environ.get("PORT", 8000))

if open_api_key is None or book_path is None:
    raise ValueError(
        "Incorrect environment variables, make sure to set OPENAI_KEY and BOOK_PATH"
    )

logging.basicConfig(level=logging.INFO)


app = FastAPI()

file_loader = FileDataLoader(book_path)
text_processor_simple = TextPreprocessorSimple()
chat_bot_simple = ChatBotServiceSimilarity(text_processor_simple, file_loader).process()
chat_bot_transformer = ChatBotServiceSentenceTransformer(
    OpenAiService(open_api_key), file_loader
).process(text_processor_simple)
chat_bot_langchain = ChatBotLangchain(book_path, open_api_key).process()


@app.post("/chat-simple", response_model=Answer)
async def chat_simple(request: QuestionReqeust):
    return Answer(answer=chat_bot_simple.get_answer(request.question))


@app.post("/chat-sentence-transformer", response_model=Answer)
async def chat_sentence_transformer(request: QuestionReqeust):
    return Answer(answer=await chat_bot_transformer.get_answer(request.question))

@app.post("/chat-langchain", response_model=Answer)
async def chat_openai(request: QuestionReqeust):
    return Answer(answer=chat_bot_langchain.get_answer(request.question))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=PORT)
