import os
import asyncio
import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# --- CONFIGURATION ---
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://router.huggingface.co/models/"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# THE UPGRADED 2026 SWARM
MODELS = {
    "supervisor": "google/gemma-4-31b-it",             # Gemma 4 (The Leader)
    "logic": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", # DeepSeek R1 (The Thinker)
    "audit": "meta-llama/Llama-3.2-3B-Instruct"         # GPT-OSS (The Auditor)
}

app = FastAPI()

class Query(BaseModel):
    prompt: str

# --- DIRECT-WIRE WORKER ---
async def call_hf(prompt, model_key, tokens):
    model_id = MODELS[model_key]
    # DeepSeek R1 works best with a thinking prompt
    formatted_prompt = f"<|thought|>\n{prompt}" if "logic" in model_key else prompt
    
    payload = {
        "inputs": formatted_prompt,
        "parameters": {"max_new_tokens": tokens, "wait_for_model": True}
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(f"{API_URL}{model_id}", json=payload, headers=HEADERS)
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', 'No response')
            elif isinstance(result, dict) and 'generated_text' in result:
                return result['generated_text']
            else:
                return f"System Note: {str(result)}"
        except Exception as e:
            return f"Swarm Error: {str(e)}"

# --- ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("index.html", "r") as f:
        return f.read()

@app.post("/api/auraflux")
async def auraflux_engine(query: Query):
    # Parallel dispatch to Specialist Brains
    l_task = call_hf(f"Step-by-step logic for: {query.prompt}", "logic", 1024)
    a_task = call_hf(f"Audit this for errors: {query.prompt}", "audit", 512)
    l_res, a_res = await asyncio.gather(l_task, a_task)
    
    # Gemma 4 Final Synthesis
    sup_prompt = f"Logic Thought Process: {l_res}\nAuditor Report: {a_res}\nUser Goal: {query.prompt}\nFinal Sovereign Consensus:"
    final = await call_hf(sup_prompt, "supervisor", 1024)
    
    return {"logic": l_res, "audit": a_res, "final": final}
