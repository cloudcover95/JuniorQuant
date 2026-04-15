import mlx.core as mx
import numpy as np
import os
import pandas as pd
import pyarrow.parquet as pq

class StreamIngestor:
    """Streams Big Data files in deterministic hardware-safe chunks to prevent O(N^3) throttling."""
    
    # Lock chunk size to 512x512 to guarantee flatline thermal load
    CHUNK_DIM = 512
    CHUNK_BYTES = CHUNK_DIM * CHUNK_DIM 

    @staticmethod
    def stream_file(file_path: str):
        """Generator that yields Mx tensors sequentially without loading the full file into RAM."""
        ext = file_path.split('.')[-1].lower()
        
        try:
            if ext in ['iso', 'bin', 'dat', 'raw', 'txt', 'md']:
                mode = 'rb' if ext in ['iso', 'bin', 'dat'] else 'r'
                encoding = None if mode == 'rb' else 'utf-8'
                
                with open(file_path, mode, encoding=encoding, errors='ignore' if mode == 'r' else None) as f:
                    while True:
                        chunk = f.read(StreamIngestor.CHUNK_BYTES)
                        if not chunk: break
                        
                        # Handle text/byte conversions
                        if isinstance(chunk, str):
                            chunk_bytes = chunk.encode('utf-8')[:StreamIngestor.CHUNK_BYTES]
                        else:
                            chunk_bytes = chunk[:StreamIngestor.CHUNK_BYTES]
                            
                        # Pad final chunk if it doesn't perfectly fit 512x512
                        if len(chunk_bytes) < StreamIngestor.CHUNK_BYTES:
                            chunk_bytes += b'\x00' * (StreamIngestor.CHUNK_BYTES - len(chunk_bytes))
                            
                        data = np.frombuffer(chunk_bytes, dtype=np.uint8).astype(np.float32)
                        grid = data.reshape((StreamIngestor.CHUNK_DIM, StreamIngestor.CHUNK_DIM))
                        
                        # Inject microscopic entropy to prevent empty matrix SVD convergence faults
                        noise = np.random.normal(0, 0.01, (StreamIngestor.CHUNK_DIM, StreamIngestor.CHUNK_DIM)).astype(np.float32)
                        yield mx.array(grid + noise)

            # For structured tabular data, we load it but chunk the tensor derivation
            elif ext in ['csv', 'parquet']:
                if ext == 'csv':
                    df = pd.read_csv(file_path)
                else:
                    df = pq.read_table(file_path).to_pandas()
                
                num_df = df.select_dtypes(include=[np.number])
                if num_df.empty and not df.empty:
                    text_data = df.astype(str).apply(lambda x: ''.join(x), axis=1).str.cat()
                    raw_bytes = text_data.encode('utf-8', errors='ignore')
                else:
                    raw_bytes = num_df.values.flatten().tobytes()
                
                for i in range(0, len(raw_bytes), StreamIngestor.CHUNK_BYTES):
                    chunk_bytes = raw_bytes[i:i+StreamIngestor.CHUNK_BYTES]
                    if len(chunk_bytes) < StreamIngestor.CHUNK_BYTES:
                        chunk_bytes += b'\x00' * (StreamIngestor.CHUNK_BYTES - len(chunk_bytes))
                    
                    data = np.frombuffer(chunk_bytes, dtype=np.uint8).astype(np.float32)
                    grid = data.reshape((StreamIngestor.CHUNK_DIM, StreamIngestor.CHUNK_DIM))
                    noise = np.random.normal(0, 0.01, (StreamIngestor.CHUNK_DIM, StreamIngestor.CHUNK_DIM)).astype(np.float32)
                    yield mx.array(grid + noise)
                    
            else:
                raise ValueError(f"Unsupported stream format: {ext}")
                
        except Exception as e:
            raise ValueError(f"Stream Engine Failure: {str(e)}")
