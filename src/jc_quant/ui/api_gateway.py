import asyncio
import json
import os
import mlx.core as mx
from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# Import compiled IP or Vault fallback
try:
    from src.jc_quant.core.atml_modulator import IsingModulator
except ImportError:
    import sys
    sys.path.append(".secure_vault")
    from jc_quant.core.atml_modulator import IsingModulator

from src.jc_quant.bridge.cuda_q_bridge import NVQLinkBridge
from src.jc_quant.telemetry.audit_ledger import LedgerAuditSystem
from src.jc_quant.core.tensor_utils import DataIngestor
from src.jc_quant.security.gate import SecurityGate, CONFIG

app = FastAPI()
app.mount("/static", StaticFiles(directory="src/jc_quant/ui/static"), name="static")

modulator = IsingModulator()
bridge = NVQLinkBridge()
ledger = LedgerAuditSystem()
active_connections = set()

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("src/jc_quant/ui/static/index.html", "r") as f:
        return f.read()

@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)
    try:
        while True: await websocket.receive_text()
    except: active_connections.remove(websocket)

async def broadcast(data: dict):
    for conn in active_connections:
        await conn.send_text(json.dumps(data))

@app.post("/api/sandbox/inject")
async def inject_dataset(file: UploadFile = File(...)):
    # 1. Signal "Processing" to UI immediately
    await broadcast({"event": "processing_start", "filename": file.filename})

    # 2. Save file temporarily
    temp_path = f"data_lake/{file.filename}"
    with open(temp_path, "wb") as f: f.write(await file.read())

    # 3. Offload heavy math to worker thread (prevents dashboard stall)
    try:
        def compute_pipeline():
            raw_state = DataIngestor.process(temp_path)
            trust_score = SecurityGate.calculate_trust_score(raw_state)
            U, S, Vt, fds = modulator.execute_decoding_loop(raw_state)
            metrics = bridge.evaluate_efficiency(S, fds)
            
            ledger.commit_audit(trust_score, fds.item(), metrics['tensor_density'], metrics['speed_multiplier'], metrics['accuracy_multiplier'])
            return {"fds": fds.item(), "speed_x": metrics['speed_multiplier'], "trust_score": trust_score}

        results = await asyncio.to_thread(compute_pipeline)
        
        # 4. Success Broadcast
        await broadcast({
            "event": "injection_complete",
            "filename": file.filename,
            **results
        })
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)

    return {"status": "success"}
