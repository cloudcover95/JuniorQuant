import mlx.core as mx
from src.jc_quant.security.gate import CONFIG

class IsingModulator:
    """AI Control Plane for recursive quantum state calibration. Vectorized manifold operations."""
    def __init__(self):
        self.fds_threshold = mx.array(CONFIG['atml']['fds_threshold'])
        self.lambda_penalty = mx.array(CONFIG['atml']['lambda_penalty'])
        self.max_iter = CONFIG['atml']['max_iter']

    def compute_calibration_error(self, Y: mx.array, U: mx.array, S: mx.array, Vt: mx.array) -> mx.array:
        """Calculates Feature Disagreement Score (FDS) via Frobenius norm."""
        residual = Y - (U @ mx.diag(S) @ Vt)
        return mx.sqrt(mx.sum(mx.square(residual)))

    def execute_decoding_loop(self, Y: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """Riemannian optimization loop. SVD soft-thresholding: \Sigma^{(t+1)} = \max(\Sigma^{(t)} - \lambda, 0)."""
        U, S, Vt = mx.linalg.svd(Y)
        for _ in range(self.max_iter):
            fds = self.compute_calibration_error(Y, U, S, Vt)
            if fds <= self.fds_threshold:
                return U, S, Vt, fds
            S = mx.maximum(S - self.lambda_penalty, mx.zeros_like(S))
            if mx.sum(S) == 0:
                break
        return U, S, Vt, fds
