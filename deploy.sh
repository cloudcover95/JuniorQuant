#!/bin/zsh
# Virtual Environment Initialization
python3 -m venv .venv
source .venv/bin/activate
pip install mlx toml pyarrow > /dev/null 2>&1

# Execute Pipeline
python3 src/jc_quant/sandbox/injector.py

# Stage for GitHub deployment
git add .
git commit -m "feat(architecture): Initialized JuniorQuant SDK with Ising ATML and pyarrow telemetry"
echo "\n[SYSTEM] JuniorQuant architecture staged for cloudcover95 deployment. Ready for git push."
