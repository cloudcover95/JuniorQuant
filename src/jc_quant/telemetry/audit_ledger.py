import pyarrow as pa
import pyarrow.parquet as pq
import time
import os
from src.jc_quant.security.gate import SecurityGate, CONFIG

class LedgerAuditSystem:
    """High-density Parquet writing for TS telemetry and efficiency reporting."""
    def __init__(self):
        self.output_dir = os.path.abspath(CONFIG['telemetry']['ledger_path'])
        SecurityGate.verify_path(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def commit_audit(self, trust_score: float, fds: float, density: float, speed_x: float, accuracy_x: float) -> None:
        """Vectorized column mapping for high-frequency writes."""
        data = [
            pa.array([time.time()]),
            pa.array([trust_score]),
            pa.array([fds]),
            pa.array([density]),
            pa.array([speed_x]),
            pa.array([accuracy_x])
        ]
        schema = pa.schema([
            ('timestamp', pa.float64()),
            ('trust_score', pa.float64()),
            ('fds', pa.float64()),
            ('tensor_density', pa.float64()),
            ('speed_multiplier', pa.float64()),
            ('accuracy_multiplier', pa.float64())
        ])
        batch = pa.RecordBatch.from_arrays(data, schema=schema)
        table = pa.Table.from_batches([batch])
        
        filepath = os.path.join(self.output_dir, f"audit_{int(time.time() * 1000)}.parquet")
        pq.write_table(table, filepath)
