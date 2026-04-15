# src/jc_quant/telemetry/ledger_analytics.py
import pyarrow.parquet as pq
import os
import glob
from src.jc_quant.security.gate import CONFIG, SecurityGate

class AuditAnalyzer:
    """Aggregates high-density Parquet ledgers for efficiency reporting."""
    def __init__(self):
        self.ledger_path = os.path.abspath(CONFIG['telemetry']['ledger_path'])
        SecurityGate.verify_path(self.ledger_path)

    def generate_report(self) -> dict:
        files = glob.glob(os.path.join(self.ledger_path, "*.parquet"))
        if not files:
            return {"status": "No audit data found."}
            
        dataset = pq.ParquetDataset(files)
        table = dataset.read()
        
        # Compute aggregate metrics across all historical injections
        fds_mean = table.column('fds').to_numpy().mean()
        speed_mean = table.column('speed_multiplier').to_numpy().mean()
        acc_mean = table.column('accuracy_multiplier').to_numpy().mean()
        
        return {
            "total_cycles_audited": len(table),
            "mean_fds": float(fds_mean),
            "mean_speed_yield_x": float(speed_mean),
            "mean_accuracy_yield_x": float(acc_mean)
        }
