import os
import asyncio
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from huggingface_hub import AsyncInferenceClient

# --- CONFIGURATION ---
HF_TOKEN = os.getenv("HF_TOKEN")
client = AsyncInferenceClient(token=HF_TOKEN)

# High-Availability Models for 2026
MODELS = {
    "supervisor": "mistralai/Mixtral-8x7B-Instruct-v0.1", 
    "logic": "Qwen/Qwen2.5-72B-Instruct",
    "audit": "meta-llama/Llama-3.2-3B-Instruct" 
}

app = FastAPI()

class Query(BaseModel):
    prompt: str

# --- STABILIZED SOVEREIGN WORKER ---
async def call_hf(prompt, model_key, tokens):
    try:
        # Standardize on non-streaming Chat Completions
        response = await client.chat.completions.create(
            model=MODELS[model_key],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=tokens,
            stream=False  # CRITICAL: Prevents StopIteration errors
        )
        return response.choices[0].message.content
    except Exception:
        # Emergency Fallback to Text Generation
        try:
            res = await client.text_generation(
                prompt,
                model=MODELS[model_key],
                max_new_tokens=tokens
            )
            return res if isinstance(res, str) else str(res)
        except Exception as e:
            return f"Swarm Error: {str(e)}"

# --- ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("index.html", "r") as f:
        return f.read()

@app.post("/api/auraflux")
async def auraflux_engine(query: Query):
    # Parallel specialist dispatch
    l_task = call_hf(f"Logic Analysis: {query.prompt}", "logic", 512)
    a_task = call_hf(f"Audit Analysis: {query.prompt}", "audit", 512)
    l_res, a_res = await asyncio.gather(l_task, a_task)
    
    # Synthesis (Supervisor Brain)
    sup_prompt = f"Logic: {l_res}\nAudit: {a_res}\nUser Query: {query.prompt}\nFinal Consensus:"
    final = await call_hf(sup_prompt, "supervisor", 1024)
    
    return {"logic": l_res, "audit": a_res, "final": final}
