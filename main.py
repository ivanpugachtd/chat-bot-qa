from fastapi import FastAPI
from src.chatbot.chat_bot_faiss import ChatBotServiceSentenceTransformer, OpenAiService
from src.data_manager.file_loader import FileDataLoader
import logging
from dotenv import load_dotenv
from src.data_manager.models import Answer
from src.data_manager.text_preprocessor import TextPreprocessorSimple
from src.data_manager.service import ChatBotServiceSimilarity
import os

load_dotenv()
open_api_key = os.environ.get("OPENAI_KEY", None)

if open_api_key is None:
    raise ValueError("OpenAI key is not provided")

logging.basicConfig(level=logging.INFO)


app = FastAPI()

file_loader = FileDataLoader(
    "src/static/Rouling_Harry_Potter_1_Harry_Potter_and_the_Sorcerers_Stone_RuLit_Net.txt"
)
text_processor_simple = TextPreprocessorSimple()
chat_bot1 = ChatBotServiceSimilarity(text_processor_simple, file_loader).process()
chat_bot2 = ChatBotServiceSentenceTransformer(
    OpenAiService(open_api_key), file_loader
).process(text_processor_simple)


@app.get("/chat-simple", response_model=Answer)
async def chat_simple(question: str):
    return Answer(answer=chat_bot1.get_answer(question))


@app.get("/chat-sentence-transformer", response_model=Answer)
async def chat_sentence_transformer(question: str):
    return Answer(answer=await chat_bot2.get_answer(question))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
