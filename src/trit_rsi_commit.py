"""Wilder RSI band trits, side-committed per port. Pure math, no I/O.

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


def rsi(xs, period: int) -> list[float]:
    """Wilder RSI in [0, 100]; warmup values are 50.0, a flat window is 50.0."""
    if period < 1:
        raise ValueError("period must be >= 1")
    out = [50.0] * len(xs)
    if len(xs) <= period:
        return out
    d = [xs[i] - xs[i - 1] for i in range(1, len(xs))]
    up = sum(max(v, 0.0) for v in d[:period]) / period
    dn = sum(max(-v, 0.0) for v in d[:period]) / period

    def val(u: float, w: float) -> float:
        if u == 0 and w == 0:
            return 50.0
        return 100.0 if w == 0 else 100.0 - 100.0 / (1.0 + u / w)

    out[period] = val(up, dn)
    for i in range(period, len(d)):
        up = (up * (period - 1) + max(d[i], 0.0)) / period
        dn = (dn * (period - 1) + max(-d[i], 0.0)) / period
        out[i + 1] = val(up, dn)
    return out


def rsi_trits(xs, period: int = 14, lo: float = 30.0, hi: float = 70.0) -> list[int]:
    """+1 if RSI > hi (overbought), -1 if RSI < lo (oversold), else 0."""
    if not 0.0 <= lo < hi <= 100.0:
        raise ValueError("need 0 <= lo < hi <= 100")
    return [1 if r > hi else (-1 if r < lo else 0) for r in rsi(xs, period)]


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


def rsi_commit(xs, period: int = 14, lo: float = 30.0, hi: float = 70.0, port: str = "Home", prev: str = GENESIS) -> dict:
    """One commit over the RSI-trit vector: sha256(prev || pack || port:rsi)[:16]."""
    if port not in PORTS:
        raise ValueError("unknown port")
    tr = rsi_trits(xs, period, lo, hi)
    c = side_commit(prev, pack(tr), f"{port}:rsi")
    return {"port": port, "prev": prev, "trits": tr, "commit": c, "rate": RATE * sum(tr)}


def verify(rec: dict) -> bool:
    return side_commit(rec["prev"], pack(rec["trits"]), f"{rec['port']}:rsi") == rec["commit"]


if __name__ == "__main__":
    xs = [10.0, 11.0, 12.0, 13.0, 14.0, 13.0, 11.0, 9.0, 7.0, 5.0]
    r = rsi(xs, 3)
    assert r[:3] == [50.0] * 3 and r[3] == 100.0, r
    tr = rsi_trits(xs, 3)
    assert tr[3] == 1 and tr[-1] == -1, tr
    assert rsi_trits([5.0] * 8, 3) == [0] * 8
    assert all(0.0 <= v <= 100.0 for v in r)
    prev = GENESIS
    tips = {}
    for p in PORTS:
        rec = rsi_commit(xs, 3, port=p, prev=prev)
        assert verify(rec) and len(rec["commit"]) == 16
        assert not verify(dict(rec, trits=[0] * len(rec["trits"])))
        tips[p] = rec["commit"]
        prev = rec["commit"]
    assert len(set(tips.values())) == len(PORTS)
    assert not (MINT or SOCKET or LAYER_ONE) and RATE == 0.05
    print({"rsi": [round(v, 2) for v in r], "trits": tr, "tips": tips, "ok": True})
