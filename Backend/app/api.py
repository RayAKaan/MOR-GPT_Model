from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import Optional


# Pydantic Schemas
class GenerationRequest(BaseModel):
    text: str
    temperature: Optional[float] = 0.8
    max_tokens: Optional[int] = 150
    top_k: Optional[int] = 40
    top_p: Optional[float] = 0.9


class GenerationResponse(BaseModel):
    generated_text: str
    tokens_used: int
    inference_time_ms: float

# API Router
router = APIRouter()


@router.post("/generate", response_model=GenerationResponse)
async def generate(request: GenerationRequest, http_request: Request):
    """
    Receives a prompt and returns a generated text sequence.
    """
    # Access the model tester instance from the app state
    model_tester = http_request.app.state.model_tester

    try:
        # Call the actual model's generation function
        result = model_tester.generate_text(
            prompt=request.text,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
        )

        # Format the response
        return GenerationResponse(
            generated_text=result['generated_text'],
            tokens_used=result.get('num_tokens_generated', 0),
            inference_time_ms=result['generation_time'] * 1000
        )

    except Exception as e:
        print(f"❌ Error during generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_status(http_request: Request):
    """Returns the operational status of the model server."""
    model_tester = http_request.app.state.model_tester
    return {
        "status": "online",
        "model_loaded": model_tester is not None,
        "device": str(model_tester.device) if model_tester else "N/A"
    }