import mlx.core as mx
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

class DataIngestor:
    """Handles .raw, .csv, and .parquet into MLX tensors. Enforces edge limits."""
    @staticmethod
    def process(file_path: str) -> mx.array:
        ext = file_path.split('.')[-1].lower()
        data = np.array([])
        
        try:
            if ext == 'raw': 
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read(270000)
                    data = np.frombuffer(text.encode('utf-8'), dtype=np.uint8).astype(np.float32)
            
            elif ext == 'csv':
                df = pd.read_csv(file_path)
                num_df = df.select_dtypes(include=[np.number])
                if num_df.empty and not df.empty:
                    text = df.astype(str).apply(lambda x: ''.join(x), axis=1).str.cat()[:270000]
                    data = np.frombuffer(text.encode('utf-8'), dtype=np.uint8).astype(np.float32)
                else:
                    data = num_df.values.astype(np.float32)
            
            elif ext == 'parquet':
                df = pq.read_table(file_path).to_pandas()
                num_df = df.select_dtypes(include=[np.number])
                if num_df.empty and not df.empty:
                    # ROOT FIX: Fallback for Hugging Face Text-Based Parquets
                    text = df.astype(str).apply(lambda x: ''.join(x), axis=1).str.cat()[:270000]
                    data = np.frombuffer(text.encode('utf-8'), dtype=np.uint8).astype(np.float32)
                else:
                    data = num_df.values.astype(np.float32)
        except Exception as e:
            raise ValueError(f"Parsing Engine Failure: {str(e)}")

        # Failsafe for mathematically invalid grid sizes
        if data.size < 16:
            raise ValueError("Dataset insufficient for manifold mapping (Size < 16 bytes).")

        # Dimensionality constraints (Cap at 512x512 for edge efficiency)
        dim = int(np.sqrt(data.size))
        if dim > 512:
            dim = 512
            
        trimmed_data = data[:dim*dim].reshape((dim, dim))
        
        # Inject microscopic entropy to prevent SVD convergence faults on highly repetitive data
        noise = np.random.normal(0, 0.01, (dim, dim)).astype(np.float32)
        return mx.array(trimmed_data + noise)
