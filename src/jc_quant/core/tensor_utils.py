import mlx.core as mx
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import os

class DataIngestor:
    """Handles multi-format big data ingestion (.raw, .txt, .csv, .parquet, .iso, .bin, .dat)."""
    @staticmethod
    def process(file_path: str) -> mx.array:
        ext = file_path.split('.')[-1].lower()
        data = np.array([])
        
        # Edge Node Hard Limit: ~512x512 matrix to prevent memory leak and power spike
        MAX_BYTES = 270000 
        
        try:
            # 1. Unstructured Text Data
            if ext in ['raw', 'txt', 'md']: 
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read(MAX_BYTES)
                    data = np.frombuffer(text.encode('utf-8'), dtype=np.uint8).astype(np.float32)
            
            # 2. Raw Binary / Disk Images (Chunked Streaming)
            elif ext in ['iso', 'bin', 'dat']:
                with open(file_path, 'rb') as f:
                    # Stream only the required bytes to prevent RAM exhaustion
                    chunk = f.read(MAX_BYTES)
                    data = np.frombuffer(chunk, dtype=np.uint8).astype(np.float32)

            # 3. Structured CSV
            elif ext == 'csv':
                df = pd.read_csv(file_path, nrows=50000) # Row limit for memory safety
                num_df = df.select_dtypes(include=[np.number])
                if num_df.empty and not df.empty:
                    text = df.astype(str).apply(lambda x: ''.join(x), axis=1).str.cat()[:MAX_BYTES]
                    data = np.frombuffer(text.encode('utf-8', errors='ignore'), dtype=np.uint8).astype(np.float32)
                else:
                    data = num_df.values.flatten()[:MAX_BYTES].astype(np.float32)
            
            # 4. High-Density Parquet
            elif ext == 'parquet':
                df = pq.read_table(file_path).to_pandas()
                num_df = df.select_dtypes(include=[np.number])
                if num_df.empty and not df.empty:
                    text = df.astype(str).apply(lambda x: ''.join(x), axis=1).str.cat()[:MAX_BYTES]
                    data = np.frombuffer(text.encode('utf-8', errors='ignore'), dtype=np.uint8).astype(np.float32)
                else:
                    data = num_df.values.flatten()[:MAX_BYTES].astype(np.float32)
            else:
                raise ValueError(f"Unsupported format: {ext}")
                
        except Exception as e:
            raise ValueError(f"Parsing Engine Failure: {str(e)}")

        # Failsafe for mathematically invalid grid sizes
        if data.size < 16:
            raise ValueError("Dataset insufficient for manifold mapping (Size < 16 bytes).")

        # Dimensionality constraints (Strict 512x512 cap)
        dim = int(np.sqrt(data.size))
        if dim > 512:
            dim = 512
            
        trimmed_data = data[:dim*dim].reshape((dim, dim))
        
        # Inject microscopic entropy to prevent SVD convergence faults on highly repetitive data (like empty ISO headers)
        noise = np.random.normal(0, 0.01, (dim, dim)).astype(np.float32)
        return mx.array(trimmed_data + noise)
