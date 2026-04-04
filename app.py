import os
import asyncio
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from huggingface_hub import AsyncInferenceClient

# --- CONFIGURATION ---
HF_TOKEN = os.getenv("HF_TOKEN")
client = AsyncInferenceClient(token=HF_TOKEN)

MODELS = {
    "supervisor": "google/gemma-4-31b-it",
    "logic": "Qwen/Qwen3.5-397B-A17B-Instruct",
    "audit": "openai/gpt-oss-120b"
}

app = FastAPI()

class Query(BaseModel):
    prompt: str

# --- WORKER LOGIC ---
async def call_model(prompt: str, model_key: str, tokens: int):
    try:
        return await client.text_generation(
            prompt, 
            model=MODELS[model_key], 
            max_new_tokens=tokens
        )
    except Exception as e:
        return f"Error in {model_key}: {str(e)}"

# --- API ROUTE ---
@app.post("/api/auraflux")
async def auraflux_engine(query: Query):
    # 1. Parallel Specialist Dispatch
    logic_task = call_model(f"Analyze logic: {query.prompt}", "logic", 512)
    audit_task = call_model(f"Audit syntax/typos: {query.prompt}", "audit", 256)
    
    l_res, a_res = await asyncio.gather(logic_task, audit_task)
    
    # 2. Gemma 4 Supervisor Synthesis
    supervisor_prompt = f"<|think|>\nLogic: {l_res}\nAudit: {a_res}\nUser: {query.prompt}\n<channel|>"
    final_consensus = await call_model(supervisor_prompt, "supervisor", 1024)
    
    return {
        "logic": l_res,
        "audit": a_res,
        "final": final_consensus
    }

# --- FRONTEND SERVING ---
@app.get("/")
async def serve_home():
    return FileResponse("/index.html")

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")
