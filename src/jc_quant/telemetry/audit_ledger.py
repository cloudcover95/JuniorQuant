import pyarrow as pa
import pyarrow.parquet as pq
import time
import os
from src.jc_quant.security.gate import SecurityGate, CONFIG

class LedgerAuditSystem:
    """High-density Parquet writing for continuous stream telemetry."""
    def __init__(self):
        self.output_dir = os.path.abspath(CONFIG['telemetry']['ledger_path'])
        SecurityGate.verify_path(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def commit_stream_audit(self, ts: list, trust: list, fds: list, density: list, speed: list, acc: list) -> None:
        """Batch mapping for entire stream payloads to prevent SSD I/O thrashing."""
        if not ts: return

        data = [
            pa.array(ts), pa.array(trust), pa.array(fds),
            pa.array(density), pa.array(speed), pa.array(acc)
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
        
        filepath = os.path.join(self.output_dir, f"audit_stream_{int(time.time())}.parquet")
        pq.write_table(table, filepath)
