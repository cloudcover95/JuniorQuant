"""Commodity Channel Index (close-only) band trits, side-committed per port. Pure math, no I/O.

Next to src/trit_side_commit.py. Ports: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
"""
from __future__ import annotations

import hashlib

RATE = 0.05
MINT = False
SOCKET = False
LAYER_ONE = False
GENESIS = "0" * 16
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")


def cci(xs, period: int) -> list[float]:
    """(x - SMA) / (0.015 * mean absolute deviation) over a trailing window. Warmup and flat windows are 0.0."""
    if period < 2:
        raise ValueError("period must be >= 2")
    out = [0.0] * len(xs)
    for i in range(period - 1, len(xs)):
        w = xs[i - period + 1 : i + 1]
        m = sum(w) / period
        md = sum(abs(v - m) for v in w) / period
        out[i] = 0.0 if md == 0 else (xs[i] - m) / (0.015 * md)
    return out


def cci_trits(xs, period: int = 20, band: float = 100.0) -> list[int]:
    """+1 if CCI > band, -1 if CCI < -band, else 0."""
    if band <= 0:
        raise ValueError("band must be > 0")
    return [1 if v > band else (-1 if v < -band else 0) for v in cci(xs, period)]


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


def cci_commit(xs, period: int = 20, band: float = 100.0, port: str = "Home", prev: str = GENESIS) -> dict:
    """One commit over the CCI-trit vector: sha256(prev || pack || port:cci)[:16]."""
    if port not in PORTS:
        raise ValueError("unknown port")
    tr = cci_trits(xs, period, band)
    c = side_commit(prev, pack(tr), f"{port}:cci")
    return {"port": port, "prev": prev, "trits": tr, "commit": c, "rate": RATE * sum(tr)}


def verify(rec: dict) -> bool:
    return side_commit(rec["prev"], pack(rec["trits"]), f"{rec['port']}:cci") == rec["commit"]


if __name__ == "__main__":
    assert cci([5.0] * 6, 3) == [0.0] * 6
    xs = [10.0, 10.0, 10.0, 10.0, 10.0, 14.0, 10.0, 10.0, 10.0, 10.0, 6.0, 10.0]
    o = cci(xs, 5)
    assert o[:4] == [0.0] * 4 and o[5] > 100.0 and o[10] < -100.0, o
    tr = cci_trits(xs, 5)
    assert tr[5] == 1 and tr[10] == -1 and tr[4] == 0, tr
    prev = GENESIS
    tips = {}
    for p in PORTS:
        rec = cci_commit(xs, 5, port=p, prev=prev)
        assert verify(rec) and len(rec["commit"]) == 16
        assert not verify(dict(rec, trits=[0] * len(rec["trits"])))
        tips[p] = rec["commit"]
        prev = rec["commit"]
    assert len(set(tips.values())) == len(PORTS)
    assert not (MINT or SOCKET or LAYER_ONE) and RATE == 0.05
    print({"trits": tr, "tips": tips, "ok": True})
