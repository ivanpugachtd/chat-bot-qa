from fastapi import FastAPI
from src.data_manager.file_loader import FileDataLoader
import logging

from src.data_manager.models import Answer
from src.data_manager.text_preprocessor import TextPreprocessorSimple
from src.data_manager.service import ChatBotServiceOption1


logging.basicConfig(level=logging.INFO)



app = FastAPI()

file_loader = FileDataLoader(
    "src/static/Rouling_Harry_Potter_1_Harry_Potter_and_the_Sorcerers_Stone_RuLit_Net.txt"
)
text_processor_simple = TextPreprocessorSimple()
chat_bot1 = ChatBotServiceOption1(text_processor_simple, file_loader).process()


@app.get("/chat-simple", response_model=Answer)
async def root(question: str):
    return Answer(answer=chat_bot1.get_answer(question))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
