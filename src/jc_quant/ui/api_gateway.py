import asyncio
import json
import os
import logging
import mlx.core as mx
from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

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
    """Thread-safe synchronous broadcast for the compute thread."""
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
            alpha = 0.1 # Exponential Moving Average weight
            chunk_count = 0

            # Iterate over the binary continuous stream
            for chunk_tensor in StreamIngestor.stream_file(temp_path):
                chunk_count += 1
                trust_score = SecurityGate.calculate_trust_score(chunk_tensor)
                
                # Math execution isolated to the 512x512 block
                U, S, Vt, fds = modulator.execute_decoding_loop(chunk_tensor)
                metrics = bridge.evaluate_efficiency(S, fds)
                
                current_fds = fds.item()
                current_speed = metrics['speed_multiplier']

                # Initialize EWMA on first chunk, otherwise apply smoothing
                if chunk_count == 1:
                    global_fds_ewma = current_fds
                    global_speed_ewma = current_speed
                else:
                    global_fds_ewma = (alpha * current_fds) + ((1 - alpha) * global_fds_ewma)
                    global_speed_ewma = (alpha * current_speed) + ((1 - alpha) * global_speed_ewma)

                # Broadcast live updates to Omni Globe per chunk
                broadcast_sync(loop, {
                    "event": "stream_update",
                    "chunk": chunk_count,
                    "fds": global_fds_ewma,
                    "speed_x": global_speed_ewma,
                    "trust_score": trust_score
                })

                # Audit Ledger Write per chunk
                ledger.commit_audit(trust_score, current_fds, metrics['tensor_density'], current_speed, metrics['accuracy_multiplier'])

            # Final sequence termination
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

    # Dispatch stream operation to background thread
    asyncio.to_thread(compute_stream)
    return {"status": "stream_initialized"}
