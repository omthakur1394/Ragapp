from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from rag_chain import bulider_rag_chain
from langchain_core.messages import HumanMessage, AIMessage

app = FastAPI(title="RAG API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


rag = bulider_rag_chain()


sessions = {}

class ChatRequest(BaseModel):
    session_id: str
    question: str

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if req.session_id not in sessions:
        sessions[req.session_id] = []

    chat_history = sessions[req.session_id]

    response = rag.invoke({
        "input": req.question,
        "chat_history": chat_history
    })


    chat_history.append(HumanMessage(content=req.question))
    chat_history.append(AIMessage(content=response))

    return {"answer": response}
