import os
import toml
import mlx.core as mx

with open("config.toml", "r") as f:
    CONFIG = toml.load(f)

class SecurityGate:
    """Implements Trusted Access for Cyber (TAC) protocols and path isolation."""
    @staticmethod
    def verify_path(target_path: str) -> None:
        abs_path = os.path.abspath(target_path)
        for forbidden in CONFIG['security']['forbidden_paths']:
            if forbidden in abs_path:
                raise PermissionError(f"[FATAL] TAC Breach. Access to {forbidden} is strictly isolated.")

    @staticmethod
    def calculate_trust_score(data_tensor: mx.array) -> float:
        """Calculates Qlik-style data trust score based on tensor structural variance."""
        variance = mx.var(data_tensor).item()
        return 1.0 / (1.0 + variance)
