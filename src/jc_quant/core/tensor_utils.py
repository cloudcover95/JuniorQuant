import mlx.core as mx
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

class DataIngestor:
    """Handles .raw (WikiText), .csv, and .parquet into MLX tensors."""
    @staticmethod
    def process(file_path: str) -> mx.array:
        ext = file_path.split('.')[-1].lower()
        
        if ext == 'raw': # WikiText Handling
            with open(file_path, 'r', encoding='utf-8') as f:
                # Basic ASCII tokenization proxy for manifold analysis
                text = f.read(50000) # Read first 50k chars for edge speed
                data = np.frombuffer(text.encode('utf-8'), dtype=np.uint8).astype(np.float32)
        elif ext == 'csv':
            df = pd.read_csv(file_path).select_dtypes(include=[np.number])
            data = df.values.astype(np.float32)
        elif ext == 'parquet':
            data = pq.read_table(file_path).to_pandas().select_dtypes(include=[np.number]).values.astype(np.float32)
        else:
            raise ValueError(f"Unsupported format: {ext}")

        # Reshape into a square-ish matrix for SVD manifold processing
        dim = int(np.sqrt(data.size))
        trimmed_data = data[:dim*dim].reshape((dim, dim))
        return mx.array(trimmed_data)
