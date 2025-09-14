from fastapi import FastAPI
from model.test_model import MoRModelTester
from . import api

app = FastAPI(title="Mini-GPT MLOps API")


@app.on_event("startup")
def load_model():
    """Load the model at startup"""
    print("🚀 Loading MoR Model...")
    # This makes the model tester instance available across the application
    app.state.model_tester = MoRModelTester()
    print("✅ Model loaded and ready.")

# Include the router from api.py
app.include_router(api.router, prefix="/api"))
