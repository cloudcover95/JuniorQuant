"""Deadband trit windows over a quant series, side-committed per port. Pure math, no I/O.

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


def deadband_trits(xs, band: float) -> list[int]:
    """Step-to-step moves: +1 above band, -1 below -band, 0 inside. First step is 0."""
    if band < 0:
        raise ValueError("band must be >= 0")
    out = []
    for i, x in enumerate(xs):
        d = 0.0 if i == 0 else x - xs[i - 1]
        out.append(1 if d > band else (-1 if d < -band else 0))
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


def windows(trits, size: int, step: int) -> list[list[int]]:
    if size <= 0 or step <= 0:
        raise ValueError("size and step must be > 0")
    return [list(trits[i:i + size]) for i in range(0, max(len(trits) - size, 0) + 1, step)]


def window_chain(xs, band: float, size: int, step: int, port: str = "Home", prev: str = GENESIS) -> dict:
    """Chain each trit window: commit = sha256(prev || pack || port:i)[:16]."""
    if port not in PORTS:
        raise ValueError("unknown port")
    links = []
    for i, w in enumerate(windows(deadband_trits(xs, band), size, step)):
        c = side_commit(prev, pack(w), f"{port}:{i}")
        links.append({"prev": prev, "trits": w, "commit": c, "rate": RATE * sum(w)})
        prev = c
    return {"port": port, "links": links, "tip": prev, "rate_sum": RATE * sum(sum(k["trits"]) for k in links)}


def verify(chain: dict, prev: str = GENESIS) -> bool:
    for i, k in enumerate(chain["links"]):
        if k["prev"] != prev or side_commit(prev, pack(k["trits"]), f"{chain['port']}:{i}") != k["commit"]:
            return False
        prev = k["commit"]
    return prev == chain["tip"]


if __name__ == "__main__":
    xs = [100.0, 100.4, 100.1, 99.5, 99.6, 100.3, 100.2, 101.0]
    tr = deadband_trits(xs, 0.15)
    assert tr == [0, 1, -1, -1, 0, 1, 0, 1]
    assert windows(tr, 4, 2) == [[0, 1, -1, -1], [-1, -1, 0, 1], [0, 1, 0, 1]]
    tips = {}
    for p in PORTS:
        ch = window_chain(xs, 0.15, 4, 2, p)
        assert verify(ch) and len(ch["tip"]) == 16
        bad = {**ch, "links": [dict(k) for k in ch["links"]]}
        bad["links"][1]["trits"] = [1, 1, 1, 1]
        assert not verify(bad)
        tips[p] = ch["tip"]
    assert len(set(tips.values())) == len(PORTS)
    hc = window_chain(xs, 0.15, 4, 2)
    assert [round(k["rate"], 12) for k in hc["links"]] == [-0.05, -0.05, 0.1]
    assert abs(hc["rate_sum"]) < 1e-12
    assert not (MINT or SOCKET or LAYER_ONE) and RATE == 0.05
    print({"trits": tr, "tips": tips, "ok": True})
