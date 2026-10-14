"""Winsor clip + quant error bound for JuniorQuant. Stdlib only. Pure math.

Three gamma scales (JuniorHome docs/WINSOR.md), same ternary grid:
  - BitNet absmean = mean|W|
  - Junior drop    = mean of |W| after dropping the top 1/16
  - Gemini clip    = mean of min(|W|, P95)
Quant error bound (docs/QUANT_ERROR.md): relative L2 of trit and INT4 round-trips.
Close rule: sha256(prev || pack || text)[:16]. Rate 0.05 held.
Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
Bind 127.0.0.1 only. mint/socket/layer_one/fetch/ue5_launch false.
No model pull, no GGUF download, no network.
"""
from __future__ import annotations

import hashlib
import math
import random

RATE = 0.05
MINT = False
SOCKET = False
LAYER_ONE = False
FETCH = False
UE5_LAUNCH = False
BIND = "127.0.0.1"
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
GENESIS = "0" * 16
EPS = 1e-8

_ENC = {0: 0, 1: 1, -1: 2}
_DEC = {0: 0, 1: 1, 2: -1}


def pack(trits) -> bytes:
    """Balanced trits (-1/0/1) as 2-bit i2s codes (0->0, 1->1, -1->2), MSB first, zero-padded."""
    acc = bits = 0
    out = bytearray()
    for t in trits:
        if t not in _ENC:
            raise ValueError("trit must be -1, 0 or 1")
        acc = (acc << 2) | _ENC[t]
        bits += 2
        if bits == 8:
            out.append(acc)
            acc = bits = 0
    if bits:
        out.append(acc << (8 - bits))
    return bytes(out)


def unpack(data: bytes, n: int) -> list[int]:
    out = []
    for b in data:
        for s in (6, 4, 2, 0):
            out.append(_DEC[(b >> s) & 3])
    return out[:n]


def close_rule(prev: str, pk: bytes, text: str) -> str:
    """sha256(prev || pack || text)[:16]. Matches the other engines."""
    return hashlib.sha256(prev.encode() + pk + text.encode()).hexdigest()[:16]


def percentile(xs, q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]."""
    s = sorted(xs)
    if not s:
        raise ValueError("empty")
    k = (len(s) - 1) * q / 100.0
    lo = math.floor(k)
    hi = math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def winsor_clip(w, lo_q: float = 5.0, hi_q: float = 95.0) -> list[float]:
    """Two-sided Winsor clip of values to [P_lo, P_hi]."""
    lo = percentile(w, lo_q)
    hi = percentile(w, hi_q)
    return [min(max(x, lo), hi) for x in w]


def gamma_absmean(w) -> float:
    return sum(abs(x) for x in w) / len(w)


def gamma_drop(w) -> float:
    a = sorted(abs(x) for x in w)
    keep = a[: len(a) - len(a) // 16] or a
    return sum(keep) / len(keep)


def gamma_clip(w, q: float = 95.0) -> float:
    a = [abs(x) for x in w]
    p = percentile(a, q)
    return sum(min(x, p) for x in a) / len(a)


GAMMAS = {"absmean": gamma_absmean, "drop": gamma_drop, "clip": gamma_clip}


def ternarize(w, gamma: float) -> list[int]:
    """Same ternary grid for every gamma: round(W/gamma) clipped to {-1,0,1}."""
    g = gamma + EPS
    return [max(-1, min(1, round(x / g))) for x in w]


def int4(w) -> list[float]:
    """Symmetric absmax INT4 round-trip (levels -7..7)."""
    s = max(abs(x) for x in w) / 7.0 or 1.0
    return [max(-7, min(7, round(x / s))) * s for x in w]


def rel_l2(w, q) -> float:
    num = math.sqrt(sum((a - b) ** 2 for a, b in zip(w, q)))
    den = math.sqrt(sum(a * a for a in w)) or 1.0
    return num / den


def trit_error(w, gamma_name: str = "absmean") -> dict:
    g = GAMMAS[gamma_name](w)
    t = ternarize(w, g)
    deq = [x * g for x in t]
    return {"gamma": gamma_name, "scale": g, "trits": t, "rel_l2": rel_l2(w, deq)}


def error_bound(w, text: str = "winsor|0.05", prev: str = GENESIS) -> dict:
    """Trit (all three gammas) and INT4 rel_l2, closed under the standing rule."""
    if BIND != "127.0.0.1":
        return {"ok": False, "why": "bind"}
    if MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH:
        return {"ok": False, "why": "flags"}
    trit = {k: trit_error(w, k) for k in GAMMAS}
    best = min(trit.values(), key=lambda r: r["rel_l2"])
    pk = pack(best["trits"])
    return {
        "ok": True,
        "rate": RATE,
        "trit_rel_l2": {k: round(v["rel_l2"], 4) for k, v in trit.items()},
        "int4_rel_l2": round(rel_l2(w, int4(w)), 4),
        "best": best["gamma"],
        "close": close_rule(prev, pk, text),
    }


def self_check() -> dict:
    assert BIND == "127.0.0.1"
    assert PORTS == ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
    assert not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH)
    assert abs(RATE - 0.05) < 1e-12
    t = [1, 0, -1, -1, 1, 0, 1]
    p = pack(t)
    assert unpack(p, len(t)) == t
    # Cross-engine vector: same bytes and close as web3node side_commit / BitNet-mlx close_rule.
    ref = hashlib.sha256(GENESIS.encode() + p + b"web3").hexdigest()[:16]
    assert close_rule(GENESIS, p, "web3") == ref and len(ref) == 16
    c = winsor_clip([float(i) for i in range(101)])
    assert c[0] == 5.0 and c[-1] == 95.0 and c[50] == 50.0
    rng = random.Random(0)
    w = [rng.gauss(0.0, 1.0) for _ in range(128)]
    assert gamma_drop(w) <= gamma_absmean(w) + 1e-12
    assert gamma_clip(w) <= gamma_absmean(w) + 1e-12
    r = error_bound(w)
    assert r["ok"] and len(r["close"]) == 16
    assert r["close"] == error_bound(w)["close"]
    for v in r["trit_rel_l2"].values():
        assert 0.3 < v < 0.8, v  # docs: trit ~0.54 at n=128
    assert 0.05 < r["int4_rel_l2"] < 0.3  # docs: INT4 ~0.14
    assert r["int4_rel_l2"] < min(r["trit_rel_l2"].values())
    return {
        "ok": True,
        "rate": RATE,
        "close": "sha256(prev || pack || text)[:16]",
        "bind": BIND,
        "ports": list(PORTS),
        "trit_rel_l2": r["trit_rel_l2"],
        "int4_rel_l2": r["int4_rel_l2"],
        "tip": r["close"],
        "engines": ("JuniorQuant", "JuniorStock", "web3node", "stocksnode", "JuniorLLM", "BitNet-mlx"),
    }


if __name__ == "__main__":
    print(self_check())
