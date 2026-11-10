"""Bollinger %B band trits, side-committed per port. Pure math, no I/O.

Next to src/trit_side_commit.py. Ports: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
"""
from __future__ import annotations

import hashlib
import math

RATE = 0.05
MINT = False
SOCKET = False
LAYER_ONE = False
GENESIS = "0" * 16
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")


def percent_b(xs, period: int = 20, k: float = 2.0) -> list[float]:
    """%B = (x - lower) / (upper - lower) over a trailing window of `period` (population std).

    Warmup and zero-width bands give 0.5 (mid-band).
    """
    if period < 2:
        raise ValueError("period must be >= 2")
    if k <= 0:
        raise ValueError("k must be > 0")
    out = [0.5] * len(xs)
    for i in range(period - 1, len(xs)):
        w = xs[i - period + 1 : i + 1]
        m = sum(w) / period
        sd = math.sqrt(sum((v - m) ** 2 for v in w) / period)
        if sd > 0.0:
            out[i] = (xs[i] - (m - k * sd)) / (2.0 * k * sd)
    return out


def bollinger_trits(xs, period: int = 20, k: float = 2.0, hi: float = 1.0, lo: float = 0.0) -> list[int]:
    """+1 if %B > hi (above upper band), -1 if %B < lo (below lower band), else 0."""
    if not lo < hi:
        raise ValueError("need lo < hi")
    return [1 if v > hi else (-1 if v < lo else 0) for v in percent_b(xs, period, k)]


def pack(trits) -> bytes:
    """Same layout as src.trit_side_commit.pack_bytes: 4-byte n || i2s bits (t+1), MSB first."""
    bits = n = 0
    for t in trits:
        bits = (bits << 2) | ((1 if t > 0 else (-1 if t < 0 else 0)) + 1)
        n += 1
    return n.to_bytes(4, "big") + bits.to_bytes((2 * n + 7) // 8 or 1, "big")


def side_commit(prev: str, pk: bytes, text: str) -> str:
    """sha256(prev || pack || text)[:16]."""
    return hashlib.sha256(prev.encode() + pk + text.encode()).hexdigest()[:16]


def bollinger_commit(xs, period: int = 20, k: float = 2.0, port: str = "Home", prev: str = GENESIS) -> dict:
    """One commit over the %B-trit vector: sha256(prev || pack || port:bollinger)[:16]."""
    if port not in PORTS:
        raise ValueError("unknown port")
    tr = bollinger_trits(xs, period, k)
    c = side_commit(prev, pack(tr), f"{port}:bollinger")
    return {"port": port, "prev": prev, "trits": tr, "commit": c, "rate": RATE * sum(tr)}


def verify(rec: dict) -> bool:
    return side_commit(rec["prev"], pack(rec["trits"]), f"{rec['port']}:bollinger") == rec["commit"]


if __name__ == "__main__":
    assert percent_b([5.0] * 6, 3) == [0.5] * 6
    xs = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 30.0, 10.0, 10.0, 10.0, 10.0, -10.0]
    tr = bollinger_trits(xs, 5, 1.5)
    assert tr[9] == 1 and tr[-1] == -1, tr
    assert tr[:4] == [0] * 4
    pb = percent_b(xs, 5, 2.0)
    prev = GENESIS
    tips = {}
    for p in PORTS:
        rec = bollinger_commit(xs, 5, 1.5, port=p, prev=prev)
        assert verify(rec) and len(rec["commit"]) == 16
        assert not verify(dict(rec, trits=[0] * len(rec["trits"])))
        tips[p] = rec["commit"]
        prev = rec["commit"]
    assert len(set(tips.values())) == len(PORTS)
    assert not (MINT or SOCKET or LAYER_ONE) and RATE == 0.05
    print({"trits": tr, "tips": tips, "ok": True})
