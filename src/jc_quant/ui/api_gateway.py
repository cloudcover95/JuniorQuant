import asyncio
import json
import os
import logging
import mlx.core as mx
from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from starlette.middleware.base import BaseHTTPMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s [API GATEWAY] %(message)s")

try:
    from src.jc_quant.core.atml_modulator import IsingModulator
except ImportError:
    import sys
    sys.path.append(".secure_vault")
    from jc_quant.core.atml_modulator import IsingModulator

from src.jc_quant.bridge.cuda_q_bridge import NVQLinkBridge
from src.jc_quant.telemetry.audit_ledger import LedgerAuditSystem
from src.jc_quant.core.tensor_utils import StreamIngestor
from src.jc_quant.security.gate import SecurityGate

app = FastAPI()

# ROOT FIX 2: Network-Level CSP to override Brave Browser Shields
class CSPMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdnjs.cloudflare.com; style-src 'self' 'unsafe-inline'; connect-src 'self' ws: wss:; img-src 'self' data:;"
        return response

app.add_middleware(CSPMiddleware)
app.mount("/static", StaticFiles(directory="src/jc_quant/ui/static"), name="static")

modulator = IsingModulator()
bridge = NVQLinkBridge()
ledger = LedgerAuditSystem()
active_connections = set()

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("src/jc_quant/ui/static/index.html", "r") as f: return f.read()

@app.websocket("/ws/telemetry")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.add(websocket)
    try:
        while True: await websocket.receive_text()
    except: active_connections.remove(websocket)

def broadcast_sync(loop, data: dict):
    async def _broadcast():
        for conn in list(active_connections):
            try: await conn.send_text(json.dumps(data))
            except: pass
    asyncio.run_coroutine_threadsafe(_broadcast(), loop)

@app.post("/api/sandbox/inject")
async def inject_dataset(file: UploadFile = File(...)):
    loop = asyncio.get_running_loop()
    broadcast_sync(loop, {"event": "processing_start", "filename": file.filename})

    os.makedirs("data_lake/parquet", exist_ok=True)
    temp_path = f"data_lake/{file.filename}"
    
    with open(temp_path, "wb") as f: 
        f.write(await file.read())

    def compute_stream():
        try:
            global_fds_ewma = 0.0
            global_speed_ewma = 0.0
            alpha = 0.1 
            chunk_count = 0

            for chunk_tensor in StreamIngestor.stream_file(temp_path):
                chunk_count += 1
                trust_score = SecurityGate.calculate_trust_score(chunk_tensor)
                
                U, S, Vt, fds = modulator.execute_decoding_loop(chunk_tensor)
                metrics = bridge.evaluate_efficiency(S, fds)
                
                current_fds = fds.item()
                current_speed = metrics['speed_multiplier']

                if chunk_count == 1:
                    global_fds_ewma = current_fds
                    global_speed_ewma = current_speed
                else:
                    global_fds_ewma = (alpha * current_fds) + ((1 - alpha) * global_fds_ewma)
                    global_speed_ewma = (alpha * current_speed) + ((1 - alpha) * global_speed_ewma)

                broadcast_sync(loop, {
                    "event": "stream_update",
                    "chunk": chunk_count,
                    "fds": global_fds_ewma,
                    "speed_x": global_speed_ewma,
                    "trust_score": trust_score
                })

                ledger.commit_audit(trust_score, current_fds, metrics['tensor_density'], current_speed, metrics['accuracy_multiplier'])

            broadcast_sync(loop, {
                "event": "injection_complete",
                "filename": file.filename,
                "fds": global_fds_ewma,
                "speed_x": global_speed_ewma
            })
            
        except Exception as e:
            logging.error(f"Stream Thread Crash: {str(e)}")
            broadcast_sync(loop, {"event": "error", "message": str(e)})
        finally:
            if os.path.exists(temp_path): os.remove(temp_path)

    # ROOT FIX 1: Explicitly wrap the coroutine in a Background Task to prevent GC execution drop
    asyncio.create_task(asyncio.to_thread(compute_stream))
    
    return {"status": "stream_initialized"}
