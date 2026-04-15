# src/jc_quant/ui/api_gateway.py
from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import mlx.core as mx
import asyncio
import json
import os

from src.jc_quant.core.atml_modulator import IsingModulator
from src.jc_quant.bridge.cuda_q_bridge import NVQLinkBridge
from src.jc_quant.telemetry.audit_ledger import LedgerAuditSystem
from src.jc_quant.telemetry.ledger_analytics import AuditAnalyzer
from src.jc_quant.security.gate import SecurityGate

app = FastAPI(title="JuniorQuant Control Plane")
app.mount("/static", StaticFiles(directory="src/jc_quant/ui/static"), name="static")

modulator = IsingModulator()
bridge = NVQLinkBridge()
ledger = LedgerAuditSystem()
analyzer = AuditAnalyzer()
active_connections = set()

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    with open("src/jc_quant/ui/static/index.html", "r") as f:
        return f.read()

@app.get("/api/audit/report")
async def get_audit_report():
    return analyzer.generate_report()

@app.post("/api/sandbox/inject")
async def inject_dataset(file: UploadFile = File(...)):
    """Simulates user import of quantum state datasets."""
    # Enforce zero-trust on filename/path
    SecurityGate.verify_path(file.filename)
    
    # Emulate ingestion and tensor conversion
    raw_state = mx.random.normal((512, 512)) # Proxy for parsed file data
    trust_score = SecurityGate.calculate_trust_score(raw_state)
    
    # ATML Modulator pipeline
    U, S, Vt, fds = modulator.execute_decoding_loop(raw_state)
    metrics = bridge.evaluate_efficiency(S, fds)
    
    # Audit logging
    ledger.commit_audit(
        trust_score, 
        fds.item(), 
        metrics['tensor_density'], 
        metrics['speed_multiplier'], 
        metrics['accuracy_multiplier']
    )
    
    payload = {
        "event": "injection_complete",
        "filename": file.filename,
        "trust_score": trust_score,
        "fds": float(fds.item()),
        "speed_x": metrics['speed_multiplier']
    }
    
    # Broadcast to iPad Dashboard
    for conn in active_connections:
        await conn.send_text(json.dumps(payload))
        
    return payload

@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)
    try:
        while True:
            await websocket.receive_text() # Keep alive
    except:
        active_connections.remove(websocket)
