import os
import asyncio
import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# --- CONFIGURATION ---
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://router.huggingface.co/models/"
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "x-wait-for-model": "true", # Crucial for 2026 high-traffic models
    "x-use-cache": "false"      # Ensures fresh reasoning for every flux
}

# THE HIGH-AVAILABILITY 2026 SWARM
MODELS = {
    "supervisor": "google/gemma-4-E9B-it",             # Gemma 4 Edge (Teacher-Distilled)
    "logic": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", # R1 Reasoning (Fast Logic)
    "audit": "meta-llama/Llama-3.2-3B-Instruct"         # GPT-OSS (Instant Audit)
}

app = FastAPI()

class Query(BaseModel):
    prompt: str

# --- HARDENED ASYNC WORKER ---
async def call_hf(prompt, model_key, tokens):
    model_id = MODELS[model_key]
    # DeepSeek R1 requires the <think> trigger to activate reasoning
    formatted_prompt = f"<think>\n{prompt}" if "logic" in model_key else prompt
    
    payload = {
        "inputs": formatted_prompt,
        "parameters": {"max_new_tokens": tokens, "return_full_text": False}
    }
    
    async with httpx.AsyncClient(timeout=45.0) as client:
        try:
            response = await client.post(f"{API_URL}{model_id}", json=payload, headers=HEADERS)
            
            # Catch HTML error pages from overloaded servers
            if response.status_code != 200:
                return f"Server Busy (Code {response.status_code}). Retrying..."
            
            result = response.json()
            if isinstance(result, list):
                return result[0].get('generated_text', 'No data.')
            return result.get('generated_text', 'Processing...')
            
        except Exception as e:
            return f"Offline: {str(e)}"

# --- ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("index.html", "r") as f:
        return f.read()

@app.post("/api/auraflux")
async def auraflux_engine(query: Query):
    # Run Logic and Audit in parallel to save Vercel execution time
    l_task = call_hf(query.prompt, "logic", 512)
    a_task = call_hf(query.prompt, "audit", 256)
    l_res, a_res = await asyncio.gather(l_task, a_task)
    
    # Final Consensus by Gemma 4 Edge
    sup_prompt = f"Logic: {l_res}\nAudit: {a_res}\nUser Intent: {query.prompt}\nSovereign Consensus:"
    final = await call_hf(sup_prompt, "supervisor", 512)
    
    return {"logic": l_res, "audit": a_res, "final": final}
