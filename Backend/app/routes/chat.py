# 📦 app/routes/chat.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from model.test_model import generate_response

router = APIRouter()

class ChatRequest(BaseModel):
    query: str
    debug: bool = False  # Optional flag to get routing internals

class ChatResponse(BaseModel):
    response: str
    entropy: float | None = None
    expected_recursion_steps: float | None = None

@router.post("/chat", response_model=ChatResponse)
def chat_with_bot(request: ChatRequest):
    try:
        result = generate_response(request.query, debug=request.debug)

        if isinstance(result, dict):
            return ChatResponse(
                response=result["response"],
                entropy=result.get("entropy"),
                expected_recursion_steps=result.get("expected_steps")
            )
        else:
            return ChatResponse(response=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
