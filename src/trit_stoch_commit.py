"""Stochastic %K band trits, side-committed per port. Pure math, no I/O.

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


def stoch_k(xs, period: int) -> list[float]:
    """%K in [0, 100] over a trailing close-only window; warmup and flat windows are 50.0."""
    if period < 1:
        raise ValueError("period must be >= 1")
    out = [50.0] * len(xs)
    for i in range(period - 1, len(xs)):
        w = xs[i - period + 1 : i + 1]
        lo, hi = min(w), max(w)
        out[i] = 50.0 if hi == lo else 100.0 * (xs[i] - lo) / (hi - lo)
    return out


def smooth(vs, n: int) -> list[float]:
    """Trailing simple mean over up to n values (%D when applied to %K)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    out = []
    for i in range(len(vs)):
        w = vs[max(0, i - n + 1) : i + 1]
        out.append(sum(w) / len(w))
    return out


def stoch_trits(xs, period: int = 14, d: int = 3, lo: float = 20.0, hi: float = 80.0) -> list[int]:
    """+1 if smoothed %K > hi (overbought), -1 if < lo (oversold), else 0."""
    if not 0.0 <= lo < hi <= 100.0:
        raise ValueError("need 0 <= lo < hi <= 100")
    return [1 if v > hi else (-1 if v < lo else 0) for v in smooth(stoch_k(xs, period), d)]


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


def stoch_commit(xs, period: int = 14, d: int = 3, lo: float = 20.0, hi: float = 80.0,
                 port: str = "Home", prev: str = GENESIS) -> dict:
    """One commit over the %K-trit vector: sha256(prev || pack || port:stoch)[:16]."""
    if port not in PORTS:
        raise ValueError("unknown port")
    tr = stoch_trits(xs, period, d, lo, hi)
    c = side_commit(prev, pack(tr), f"{port}:stoch")
    return {"port": port, "prev": prev, "trits": tr, "commit": c, "rate": RATE * sum(tr)}


def verify(rec: dict) -> bool:
    return side_commit(rec["prev"], pack(rec["trits"]), f"{rec['port']}:stoch") == rec["commit"]


if __name__ == "__main__":
    xs = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 13.0, 10.0, 8.0, 6.0, 5.0]
    k = stoch_k(xs, 4)
    assert k[:3] == [50.0] * 3 and k[3] == 100.0 and k[-1] == 0.0, k
    assert all(0.0 <= v <= 100.0 for v in k)
    tr = stoch_trits(xs, 4, 1)
    assert tr[3] == 1 and tr[-1] == -1, tr
    assert stoch_trits([5.0] * 8, 3) == [0] * 8
    assert smooth([1.0, 2.0, 3.0], 2) == [1.0, 1.5, 2.5]
    prev = GENESIS
    tips = {}
    for p in PORTS:
        rec = stoch_commit(xs, 4, 1, port=p, prev=prev)
        assert verify(rec) and len(rec["commit"]) == 16
        assert not verify(dict(rec, trits=[0] * len(rec["trits"])))
        tips[p] = rec["commit"]
        prev = rec["commit"]
    assert len(set(tips.values())) == len(PORTS)
    assert not (MINT or SOCKET or LAYER_ONE) and RATE == 0.05
    print({"k": [round(v, 2) for v in k], "trits": tr, "tips": tips, "ok": True})
