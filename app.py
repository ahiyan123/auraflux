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
    "x-wait-for-model": "true", # Force the API to wait for the model to load
    "Content-Type": "application/json"
}

MODELS = {
    "supervisor": "google/gemma-4-9b-it",             # Gemma 4 Edge (Fast Consensus)
    "logic": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", # R1 Logic (Fast Thinking)
    "audit": "meta-llama/Llama-3.2-3B-Instruct"         # LLAMA (Instant Audit)
}

app = FastAPI()

class Query(BaseModel):
    prompt: str

# --- ROBUST WORKER ---
async def call_hf(prompt, model_key, tokens):
    model_id = MODELS[model_key]
    # DeepSeek R1 reasoning trigger
    formatted_prompt = f"<think>\n{prompt}" if "logic" in model_key else prompt
    
    payload = {
        "inputs": formatted_prompt,
        "parameters": {"max_new_tokens": tokens}
    }
    
    async with httpx.AsyncClient(timeout=90.0) as client: # Increased timeout for heavy models
        for attempt in range(3): # 3 Retries for stability
            try:
                response = await client.post(f"{API_URL}{model_id}", json=payload, headers=HEADERS)
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list): return result[0].get('generated_text', '')
                    return result.get('generated_text', str(result))
                
                elif response.status_code == 503: # Model is loading
                    await asyncio.sleep(5)
                    continue
                    
            except Exception as e:
                if attempt == 2: return f"Swarm Error: {str(e)}"
                await asyncio.sleep(2)
    return "Error: Swarm timed out."

# --- ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("index.html", "r") as f:
        return f.read()

@app.post("/api/auraflux")
async def auraflux_engine(query: Query):
    # Execute specialists in parallel
    l_task = call_hf(query.prompt, "logic", 1024)
    a_task = call_hf(query.prompt, "audit", 512)
    l_res, a_res = await asyncio.gather(l_task, a_task)
    
    # Gemma 4 Consensus
    sup_prompt = f"Logic: {l_res}\nAudit: {a_res}\nTask: {query.prompt}\nConsensus:"
    final = await call_hf(sup_prompt, "supervisor", 1024)
    
    return {"logic": l_res, "audit": a_res, "final": final}
