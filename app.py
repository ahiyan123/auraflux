import os
import asyncio
import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# --- 2026 PERMANENT ROUTER CONFIG ---
HF_TOKEN = os.getenv("HF_TOKEN")
# This is the modern V1 endpoint that handles Gemma 4 and DeepSeek R1
API_URL = "https://router.huggingface.co/hf-inference/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
    "x-wait-for-model": "true" # Forces the server to load the model instead of 404ing
}

# THE APRIL 2026 FLAGSHIP SWARM
MODELS = {
    "supervisor": "google/gemma-4-31b-it",             # Gemma 4 (Apache 2.0)
    "logic": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", # R1 Reasoning Specialist
    "audit": "meta-llama/Llama-3.2-3B-Instruct"         # GPT-OSS (Fast Auditor)
}

app = FastAPI()

class Query(BaseModel):
    prompt: str

# --- HARDENED ROUTER WORKER ---
async def call_hf(prompt, model_key, tokens):
    model_id = MODELS[model_key]
    
    # R1 works best with an explicit 'Reasoning' request
    messages = [{"role": "user", "content": prompt}]
    
    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": tokens,
        "stream": False
    }
    
    async with httpx.AsyncClient(timeout=95.0) as client:
        try:
            response = await client.post(API_URL, json=payload, headers=HEADERS)
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            
            # Catching the transition errors specifically
            return f"Router Note (Code {response.status_code}): {response.text[:100]}"
            
        except Exception as e:
            return f"Swarm Offline: {str(e)}"

# --- ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("index.html", "r") as f:
        return f.read()

@app.post("/api/auraflux")
async def auraflux_engine(query: Query):
    # Parallel dispatch to Specialists
    l_task = call_hf(f"Analyze with deep logic: {query.prompt}", "logic", 1000)
    a_task = call_hf(f"Audit for technical errors: {query.prompt}", "audit", 512)
    l_res, a_res = await asyncio.gather(l_task, a_task)
    
    # Gemma 4 Synthesis
    sup_prompt = f"Logic: {l_res}\nAudit: {a_res}\nTask: {query.prompt}\nConsensus:"
    final = await call_hf(sup_prompt, "supervisor", 1024)
    
    return {"logic": l_res, "audit": a_res, "final": final}
