import asyncio
from fastapi import FastAPI, WebSocket
from backend.workers import call_logic_expert, call_audit_expert
from backend.supervisor import brain

app = FastAPI()

@app.websocket("/ws/auraflux")
async def auraflux_stream(websocket: WebSocket):
    await websocket.accept()
    
    while True:
        prompt = await websocket.receive_text()
        
        # Parallel Execution: The Swarm Dispatches
        logic_task = call_logic_expert(prompt)
        audit_task = call_audit_expert(prompt)
        
        l_res, a_res = await asyncio.gather(logic_task, audit_task)
        
        # Local Synthesis: Gemma 4 Judges
        final_consensus = brain.judge(prompt, l_res, a_res)
        
        await websocket.send_json({
            "logic": l_res,
            "audit": a_res,
            "final": final_consensus
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
