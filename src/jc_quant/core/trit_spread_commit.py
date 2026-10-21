"""Pair-spread z-score trits, side-committed per port. Pure math, no I/O.

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


def spread(a, b, hedge: float = 1.0) -> list[float]:
    """a[i] - hedge * b[i]."""
    if len(a) != len(b):
        raise ValueError("series lengths differ")
    return [x - hedge * y for x, y in zip(a, b)]


def zscore_trits(xs, lookback: int, k: float) -> list[int]:
    """+1 if z > k, -1 if z < -k, else 0. z uses the prior `lookback` values; warmup and flat windows are 0."""
    if lookback < 2 or k < 0:
        raise ValueError("lookback must be >= 2 and k >= 0")
    out = []
    for i, x in enumerate(xs):
        if i < lookback:
            out.append(0)
            continue
        w = xs[i - lookback:i]
        m = sum(w) / lookback
        sd = math.sqrt(sum((v - m) ** 2 for v in w) / (lookback - 1))
        z = 0.0 if sd == 0 else (x - m) / sd
        out.append(1 if z > k else (-1 if z < -k else 0))
    return out


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


def spread_commit(a, b, lookback: int, k: float, hedge: float = 1.0, port: str = "Home", prev: str = GENESIS) -> dict:
    """One commit over the whole spread-trit vector: sha256(prev || pack || port:spread)[:16]."""
    if port not in PORTS:
        raise ValueError("unknown port")
    tr = zscore_trits(spread(a, b, hedge), lookback, k)
    c = side_commit(prev, pack(tr), f"{port}:spread")
    return {"port": port, "prev": prev, "trits": tr, "commit": c, "rate": RATE * sum(tr)}


def verify(rec: dict) -> bool:
    return side_commit(rec["prev"], pack(rec["trits"]), f"{rec['port']}:spread") == rec["commit"]


if __name__ == "__main__":
    a = [10.0, 10.1, 10.0, 10.2, 10.1, 11.0, 10.1, 9.2, 10.0]
    b = [10.0] * 9
    s = spread(a, b)
    tr = zscore_trits(s, 4, 1.5)
    assert tr == [0, 0, 0, 0, 0, 1, 0, -1, 0], tr
    assert zscore_trits([1.0] * 6, 3, 1.0) == [0] * 6
    prev = GENESIS
    tips = {}
    for p in PORTS:
        r = spread_commit(a, b, 4, 1.5, port=p, prev=prev)
        assert verify(r) and len(r["commit"]) == 16
        bad = dict(r, trits=[1] * len(r["trits"]))
        assert not verify(bad)
        tips[p] = r["commit"]
        prev = r["commit"]
    assert len(set(tips.values())) == len(PORTS)
    assert abs(spread_commit(a, b, 4, 1.5)["rate"]) < 1e-12
    assert not (MINT or SOCKET or LAYER_ONE) and RATE == 0.05
    print({"trits": tr, "tips": tips, "ok": True})
