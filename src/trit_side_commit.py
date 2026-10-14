"""Trit side commit next to src/trit_unpack.py. Pure math, no I/O."""
from __future__ import annotations

import hashlib

from src.trit_unpack import pack_i2s, unpack_i2s

RATE = 0.05
MINT = False
SOCKET = False
LAYER_ONE = False


def pack_bytes(trits) -> bytes:
    bits, n = pack_i2s(trits)
    return n.to_bytes(4, "big") + bits.to_bytes((2 * n + 7) // 8 or 1, "big")


def side_commit(prev: str, pack: bytes, text: str) -> str:
    """sha256(prev || pack || text)[:16]."""
    return hashlib.sha256(prev.encode() + pack + text.encode()).hexdigest()[:16]


def rate_step(x: float) -> float:
    return x * RATE


if __name__ == "__main__":
    t = [1, 0, -1, -1, 1, 0]
    bits, n = pack_i2s(t)
    assert unpack_i2s(bits, n) == t
    p = pack_bytes(t)
    c = side_commit("0" * 16, p, "quant")
    assert len(c) == 16 and c == side_commit("0" * 16, p, "quant")
    assert not (MINT or SOCKET or LAYER_ONE) and abs(rate_step(1.0) - 0.05) < 1e-12
    print({"pack": p.hex(), "commit": c, "ok": True})
