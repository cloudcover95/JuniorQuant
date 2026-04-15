import mlx.core as mx
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import os
import math
from src.jc_quant.security.gate import CONFIG

class HardwareProfiler:
    """Profiles local Apple Silicon specs to dynamically bound tensor dimensions."""
    @staticmethod
    def calculate_manifold_horizon() -> int:
        # 1. Profile Unified Memory (macOS specific sysctl proxy via os.sysconf)
        try:
            total_ram_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
            ram_gb = total_ram_bytes / (1024 ** 3)
        except ValueError:
            ram_gb = 16.0 # Fallback for edge nodes if sysconf fails

        # 2. Evaluate Power Profile from config.toml
        power_mode = CONFIG.get('hardware', {}).get('power_profile', 'Mains_Power')
        
        # 3. Dynamic Thresholding based on O(N^3) thermal limits
        if power_mode == "48V_LiFePO4":
            # Strict thermal envelope. Scale up to 1024 if memory permits.
            base_dim = 512
            if ram_gb >= 16: base_dim = 1024
            if ram_gb >= 32: base_dim = 1536
        else:
            # Mains power (Grid/High-Performance Mode). Allow larger memory footprints.
            base_dim = 1024
            if ram_gb >= 16: base_dim = 2048
            if ram_gb >= 64: base_dim = 4096

        # Cap absolute maximum to prevent SVD convergence lockups on CPU stream
        return min(base_dim, 4096)

class DataIngestor:
    """Hardware-aware ingestion pipeline for Big Data arrays."""
    @staticmethod
    def process(file_path: str) -> mx.array:
        ext = file_path.split('.')[-1].lower()
        data = np.array([])
        
        # DYNAMIC HORIZON: Profile hardware to get max safe dimension
        MAX_DIM = HardwareProfiler.calculate_manifold_horizon()
        MAX_BYTES = MAX_DIM * MAX_DIM 
        
        try:
            if ext in ['raw', 'txt', 'md']: 
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read(MAX_BYTES)
                    data = np.frombuffer(text.encode('utf-8'), dtype=np.uint8).astype(np.float32)
            
            elif ext in ['iso', 'bin', 'dat']:
                with open(file_path, 'rb') as f:
                    chunk = f.read(MAX_BYTES)
                    data = np.frombuffer(chunk, dtype=np.uint8).astype(np.float32)

            elif ext == 'csv':
                df = pd.read_csv(file_path, nrows=100000)
                num_df = df.select_dtypes(include=[np.number])
                if num_df.empty and not df.empty:
                    text = df.astype(str).apply(lambda x: ''.join(x), axis=1).str.cat()[:MAX_BYTES]
                    data = np.frombuffer(text.encode('utf-8', errors='ignore'), dtype=np.uint8).astype(np.float32)
                else:
                    data = num_df.values.flatten()[:MAX_BYTES].astype(np.float32)
            
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

        if data.size < 16:
            raise ValueError("Dataset insufficient for manifold mapping.")

        # Dynamically calculate grid to match the ingested byte count exactly
        dim = int(np.sqrt(data.size))
        if dim > MAX_DIM:
            dim = MAX_DIM
            
        trimmed_data = data[:dim*dim].reshape((dim, dim))
        noise = np.random.normal(0, 0.01, (dim, dim)).astype(np.float32)
        return mx.array(trimmed_data + noise)
