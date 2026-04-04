import os
import asyncio
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from huggingface_hub import AsyncInferenceClient

# --- CONFIGURATION ---
# Ensure HF_TOKEN is set in Vercel Environment Variables
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

# --- CORE SWARM LOGIC ---
async def call_hf_api(prompt: str, model_key: str, tokens: int):
    try:
        return await client.text_generation(
            prompt, 
            model=MODELS[model_key], 
            max_new_tokens=tokens
        )
    except Exception as e:
        return f"Error in {model_key}: {str(e)}"

# --- ROUTES ---

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serves the dashboard from the root directory."""
    with open("index.html", "r") as f:
        return f.read()

@app.post("/api/auraflux")
async def auraflux_process(query: Query):
    # 1. Parallel Specialist Dispatch
    l_task = call_hf_api(f"Logic Analysis: {query.prompt}", "logic", 512)
    a_task = call_hf_api(f"Audit & Syntax: {query.prompt}", "audit", 256)
    
    l_res, a_res = await asyncio.gather(l_task, a_task)
    
    # 2. Gemma 4 Sovereign Synthesis
    supervisor_input = f"<|think|>\nLogic: {l_res}\nAudit: {a_res}\nUser: {query.prompt}\n<channel|>"
    final_consensus = await call_hf_api(supervisor_input, "supervisor", 1024)
    
    return {
        "logic": l_res,
        "audit": a_res,
        "final": final_consensus
    }
