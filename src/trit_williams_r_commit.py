"""Williams %R band trits, side-committed per port. Pure math, no I/O.

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


def williams_r(highs, lows, closes, period: int = 14) -> list[float]:
    """%R = -100 * (HH - close) / (HH - LL) over a trailing window; range [-100, 0].

    Warmup and zero-range windows give -50.0 (neutral).
    """
    if period < 1:
        raise ValueError("period must be >= 1")
    if not (len(highs) == len(lows) == len(closes)):
        raise ValueError("highs, lows, closes must be the same length")
    out = [-50.0] * len(closes)
    for i in range(period - 1, len(closes)):
        hh = max(highs[i - period + 1 : i + 1])
        ll = min(lows[i - period + 1 : i + 1])
        if hh > ll:
            out[i] = -100.0 * (hh - closes[i]) / (hh - ll)
    return out


def williams_trits(highs, lows, closes, period: int = 14, ob: float = -20.0, os_: float = -80.0) -> list[int]:
    """+1 if %R > ob (overbought), -1 if %R < os_ (oversold), else 0."""
    if not os_ < ob:
        raise ValueError("need os_ < ob")
    return [1 if v > ob else (-1 if v < os_ else 0) for v in williams_r(highs, lows, closes, period)]


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


def williams_commit(highs, lows, closes, period: int = 14, port: str = "Home", prev: str = GENESIS) -> dict:
    """One commit over the %R-trit vector: sha256(prev || pack || port:williams_r)[:16]."""
    if port not in PORTS:
        raise ValueError("unknown port")
    tr = williams_trits(highs, lows, closes, period)
    c = side_commit(prev, pack(tr), f"{port}:williams_r")
    return {"port": port, "prev": prev, "trits": tr, "commit": c, "rate": RATE * sum(tr)}


def verify(rec: dict) -> bool:
    return side_commit(rec["prev"], pack(rec["trits"]), f"{rec['port']}:williams_r") == rec["commit"]


if __name__ == "__main__":
    assert williams_r([5.0] * 5, [5.0] * 5, [5.0] * 5, 3) == [-50.0] * 5
    hi = [10, 11, 12, 13, 14, 15, 14, 13, 12, 11]
    lo = [8, 9, 10, 11, 12, 13, 12, 11, 10, 9]
    cl = [9, 10, 11, 12, 13, 15, 12, 11, 10, 9]
    r = williams_r(hi, lo, cl, 4)
    assert all(-100.0 <= v <= 0.0 for v in r)
    tr = williams_trits(hi, lo, cl, 4)
    assert tr[5] == 1 and tr[-1] == -1, tr
    assert tr[:3] == [0] * 3
    prev = GENESIS
    tips = {}
    for p in PORTS:
        rec = williams_commit(hi, lo, cl, 4, port=p, prev=prev)
        assert verify(rec) and len(rec["commit"]) == 16
        assert not verify(dict(rec, trits=[0] * len(rec["trits"])))
        tips[p] = rec["commit"]
        prev = rec["commit"]
    assert len(set(tips.values())) == len(PORTS)
    assert not (MINT or SOCKET or LAYER_ONE) and RATE == 0.05
    print({"trits": tr, "tips": tips, "ok": True})
