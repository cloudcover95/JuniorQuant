#!/bin/zsh
source .venv/bin/activate
echo "[SYSTEM] Initializing JuniorQuant API Gateway on 0.0.0.0:8000"
uvicorn src.jc_quant.ui.api_gateway:app --host 0.0.0.0 --port 8000
