from pydantic import BaseModel


class Answer(BaseModel):
    answer: str
    
class QuestionReqeust(BaseModel):
    question: str