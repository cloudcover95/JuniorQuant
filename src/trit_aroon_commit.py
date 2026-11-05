"""Aroon oscillator trits, side-committed per port. Pure math, no I/O.

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


def aroon_osc(xs, period: int) -> list[float]:
    """Aroon up minus Aroon down in [-100, 100] over a trailing close-only window of period+1.

    Ties resolve to the most recent high/low. Warmup values are 0.0.
    """
    if period < 1:
        raise ValueError("period must be >= 1")
    out = [0.0] * len(xs)
    for i in range(period, len(xs)):
        w = xs[i - period : i + 1]
        hi, lo = max(w), min(w)
        since_hi = period - max(j for j, v in enumerate(w) if v == hi)
        since_lo = period - max(j for j, v in enumerate(w) if v == lo)
        up = 100.0 * (period - since_hi) / period
        dn = 100.0 * (period - since_lo) / period
        out[i] = up - dn
    return out


def aroon_trits(xs, period: int = 14, band: float = 50.0) -> list[int]:
    """+1 if oscillator > band (uptrend), -1 if < -band (downtrend), else 0."""
    if not 0.0 <= band < 100.0:
        raise ValueError("need 0 <= band < 100")
    return [1 if v > band else (-1 if v < -band else 0) for v in aroon_osc(xs, period)]


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


def aroon_commit(xs, period: int = 14, band: float = 50.0, port: str = "Home", prev: str = GENESIS) -> dict:
    """One commit over the Aroon-trit vector: sha256(prev || pack || port:aroon)[:16]."""
    if port not in PORTS:
        raise ValueError("unknown port")
    tr = aroon_trits(xs, period, band)
    c = side_commit(prev, pack(tr), f"{port}:aroon")
    return {"port": port, "prev": prev, "trits": tr, "commit": c, "rate": RATE * sum(tr)}


def verify(rec: dict) -> bool:
    return side_commit(rec["prev"], pack(rec["trits"]), f"{rec['port']}:aroon") == rec["commit"]


if __name__ == "__main__":
    up = [float(i) for i in range(10)]
    o = aroon_osc(up, 4)
    assert o[:4] == [0.0] * 4 and all(v == 100.0 for v in o[4:]), o
    dn = [float(-i) for i in range(10)]
    assert all(v == -100.0 for v in aroon_osc(dn, 4)[4:])
    assert aroon_trits([5.0] * 8, 3) == [0] * 8
    xs = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 13.0, 10.0, 8.0, 6.0, 5.0]
    tr = aroon_trits(xs, 4)
    assert tr[4] == 1 and tr[-1] == -1, tr
    assert all(-100.0 <= v <= 100.0 for v in aroon_osc(xs, 4))
    prev = GENESIS
    tips = {}
    for p in PORTS:
        rec = aroon_commit(xs, 4, port=p, prev=prev)
        assert verify(rec) and len(rec["commit"]) == 16
        assert not verify(dict(rec, trits=[0] * len(rec["trits"])))
        tips[p] = rec["commit"]
        prev = rec["commit"]
    assert len(set(tips.values())) == len(PORTS)
    assert not (MINT or SOCKET or LAYER_ONE) and RATE == 0.05
    print({"osc": aroon_osc(xs, 4), "trits": tr, "tips": tips, "ok": True})
