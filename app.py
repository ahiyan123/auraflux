import os
import asyncio
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from huggingface_hub import AsyncInferenceClient

# --- CONFIGURATION ---
HF_TOKEN = os.getenv("HF_TOKEN")
client = AsyncInferenceClient(token=HF_TOKEN)

# Using stable, high-availability models for $0 budget
MODELS = {
    "supervisor": "google/gemma-2-27b-it",
    "logic": "Qwen/Qwen2.5-72B-Instruct",
    "audit": "mistralai/Mixtral-8x7B-Instruct-v0.1"
}

app = FastAPI()

class Query(BaseModel):
    prompt: str

# --- UPDATED AI WORKER (Chat Completion) ---
async def call_hf(prompt, model_key, tokens):
    try:
        # Switching to chat_completion to support Groq/Novita providers
        response = await client.chat_completion(
            model=MODELS[model_key],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"System Error: {str(e)}"

# --- ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("index.html", "r") as f:
        return f.read()

@app.post("/api/auraflux")
async def auraflux_engine(query: Query):
    # Parallel specialist dispatch
    l_task = call_hf(f"Analyze the logic of this: {query.prompt}", "logic", 512)
    a_task = call_hf(f"Audit for errors/improvements: {query.prompt}", "audit", 512)
    l_res, a_res = await asyncio.gather(l_task, a_task)
    
    # Gemma 2 Synthesis
    sup_prompt = f"Logic: {l_res}\nAudit: {a_res}\nUser Query: {query.prompt}\nProvide a final sovereign consensus."
    final = await call_hf(sup_prompt, "supervisor", 1024)
    
    return {"logic": l_res, "audit": a_res, "final": final}
