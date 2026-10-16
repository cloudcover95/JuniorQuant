"""i2sd quant-error budget: closed-form Gaussian trit error per Winsor gamma. Stdlib only.

Roadmap item: JuniorHome docs/ROADMAP.md step 3, "i2sd userspace - 127.0.0.1:8767.
No docker.sock." (Quant Chief, pair A, 2026-10-16). This file is the quant-error and
Winsor-bound side of that item: before the i2sd userspace accepts a BitNet ternary
tensor, each I2_S block (n=128) gets an error budget it must stay under.

New capability vs the earlier Winsor files: an analytic (closed-form, erf-based)
expected relative L2 for the ternary grid round(W/gamma) clipped to {-1,0,1} on a
Gaussian block, for each of the three gammas in docs/WINSOR.md:
  - BitNet absmean = mean|W|                         -> sqrt(2/pi) * sigma
  - Junior drop    = mean |W| after dropping top 1/16 -> truncated half-normal mean
  - Gemini clip    = mean of min(|W|, P95)            -> clipped half-normal mean
plus the error-optimal gamma for that grid, and a per-block admission manifest
(measured rel_l2 <= analytic + slack) closed block by block under the close rule.
Analytic absmean trit rel_l2 = 0.513 (optimal gamma 1.224 sigma gives 0.436), matching docs/QUANT_ERROR.md (~0.54 at n=128).

Close rule: sha256(prev || pack || text)[:16], pack = i2s trit codes (0->0, 1->1,
-1->2, MSB first). Rate 0.05 held. Same pack and close as the BitNet-mlx, JuniorLLM,
web3node, stocksnode, JuniorQuant and JuniorStock engines.
Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
The i2sd endpoint is named only (http://127.0.0.1:8767); no socket is opened here.
Bind 127.0.0.1 only. mint/socket/layer_one/fetch/ue5_launch false. No docker.sock.
No model pull, no GGUF download, no network.

Usage: python3 i2sd_quant_budget.py [--selftest]
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys

RATE = 0.05
MINT = False
SOCKET = False
LAYER_ONE = False
FETCH = False
UE5_LAUNCH = False
BIND = "127.0.0.1"
I2SD_PORT = 8767
I2SD_ENDPOINT = "http://%s:%d" % (BIND, I2SD_PORT)
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
ROADMAP_ITEM = "JuniorHome docs/ROADMAP.md #3 i2sd userspace 127.0.0.1:8767, no docker.sock"
GENESIS = "0" * 16
BLOCK = 128
SLACK = 0.10
EPS = 1e-8

_ENC = {0: 0, 1: 1, -1: 2}
_DEC = {0: 0, 1: 1, 2: -1}


# ---- shared engine primitives (pack + close rule) ---------------------------------

def pack(trits) -> bytes:
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


def unpack(data: bytes, n: int) -> list:
    out = []
    for b in data:
        for s in (6, 4, 2, 0):
            out.append(_DEC[(b >> s) & 3])
    return out[:n]


def close_rule(prev: str, pk: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode() + pk + text.encode()).hexdigest()[:16]


# ---- standard normal, stdlib only --------------------------------------------------

def phi(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def Phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def Phi_inv(p: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError("p in (0,1)")
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if Phi(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---- analytic gammas for a unit Gaussian (scale by sigma) --------------------------

def gamma_absmean_gauss() -> float:
    return math.sqrt(2.0 / math.pi)


def gamma_drop_gauss(drop: float = 1.0 / 16.0) -> float:
    """E[|x| | |x| < q], q the (1-drop) quantile of the half-normal."""
    q = Phi_inv(1.0 - drop / 2.0)
    return 2.0 * (phi(0.0) - phi(q)) / (1.0 - drop)


def gamma_clip_gauss(pct: float = 95.0) -> float:
    """E[min(|x|, P)], P the pct quantile of the half-normal."""
    tail = 1.0 - pct / 100.0
    p = Phi_inv(1.0 - tail / 2.0)
    return 2.0 * (phi(0.0) - phi(p)) + p * tail


GAUSS_GAMMAS = {"absmean": gamma_absmean_gauss, "drop": gamma_drop_gauss, "clip": gamma_clip_gauss}


def trit_mse_gauss(s: float) -> float:
    """E[(x - s*clip(round(x/s),-1,1))^2] for x ~ N(0,1); threshold s/2, level s."""
    a = s / 2.0
    inner = (2.0 * Phi(a) - 1.0) - 2.0 * a * phi(a)
    t = 1.0 - Phi(a)
    tail = (a * phi(a) + t) - 2.0 * s * phi(a) + s * s * t
    return inner + 2.0 * tail


def trit_rel_l2_gauss(s: float) -> float:
    return math.sqrt(trit_mse_gauss(s))


def optimal_gamma_gauss() -> float:
    """Golden-section minimum of trit_mse_gauss over s in [0.1, 3]."""
    g = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = 0.1, 3.0
    c, d = b - g * (b - a), a + g * (b - a)
    for _ in range(120):
        if trit_mse_gauss(c) < trit_mse_gauss(d):
            b = d
        else:
            a = c
        c, d = b - g * (b - a), a + g * (b - a)
    return 0.5 * (a + b)


# ---- measured, per block -----------------------------------------------------------

def percentile(xs, q: float) -> float:
    s = sorted(xs)
    k = (len(s) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


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


def ternarize(w, gamma: float) -> list:
    g = gamma + EPS
    return [max(-1, min(1, round(x / g))) for x in w]


def rel_l2(w, q) -> float:
    num = math.sqrt(sum((a - b) ** 2 for a, b in zip(w, q)))
    den = math.sqrt(sum(a * a for a in w)) or 1.0
    return num / den


def block_budget(w, variant: str) -> dict:
    """Measured trit rel_l2 for one block vs the closed-form Gaussian expectation."""
    g = GAMMAS[variant](w)
    rms = math.sqrt(sum(x * x for x in w) / len(w)) or 1.0
    t = ternarize(w, g)
    meas = rel_l2(w, [x * g for x in t])
    expect = trit_rel_l2_gauss(g / rms)
    return {"gamma": g, "trits": t, "measured": meas, "expected": expect,
            "admit": meas <= expect + SLACK}


def i2sd_manifest(weights, variant: str = "absmean", prev: str = GENESIS) -> dict:
    """Per-block admission manifest for the i2sd userspace, closed block by block."""
    if BIND != "127.0.0.1" or not I2SD_ENDPOINT.startswith("http://127.0.0.1:"):
        return {"ok": False, "why": "bind"}
    if MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH:
        return {"ok": False, "why": "flags"}
    if variant not in GAMMAS:
        return {"ok": False, "why": "variant"}
    blocks, close = [], prev
    for i in range(0, len(weights), BLOCK):
        b = block_budget(weights[i:i + BLOCK], variant)
        close = close_rule(close, pack(b["trits"]), "i2sd|%s|%d|%.2f" % (variant, i // BLOCK, RATE))
        blocks.append({"block": i // BLOCK, "gamma": round(b["gamma"], 6),
                       "measured": round(b["measured"], 4), "expected": round(b["expected"], 4),
                       "admit": b["admit"], "close": close})
    return {"ok": True, "endpoint": I2SD_ENDPOINT, "variant": variant, "rate": RATE,
            "block": BLOCK, "slack": SLACK, "admit_all": all(b["admit"] for b in blocks),
            "blocks": blocks, "close": close}


def analytic_table() -> dict:
    s_opt = optimal_gamma_gauss()
    out = {k: {"gamma": round(f(), 4), "rel_l2": round(trit_rel_l2_gauss(f()), 4)}
           for k, f in GAUSS_GAMMAS.items()}
    out["optimal"] = {"gamma": round(s_opt, 4), "rel_l2": round(trit_rel_l2_gauss(s_opt), 4)}
    return out


# ---- self-test ---------------------------------------------------------------------

def self_check() -> dict:
    assert BIND == "127.0.0.1" and I2SD_ENDPOINT == "http://127.0.0.1:8767"
    assert PORTS == ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
    assert not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH)
    assert abs(RATE - 0.05) < 1e-12
    # Cross-engine pack + close vector (same bytes as the other engines).
    t = [1, 0, -1, -1, 1, 0, 1]
    p = pack(t)
    assert p == bytes([0b01001010, 0b01000100]), p
    assert unpack(p, len(t)) == t
    assert close_rule(GENESIS, p, "web3") == hashlib.sha256(GENESIS.encode() + p + b"web3").hexdigest()[:16]
    # Normal helpers.
    assert abs(Phi(1.959964) - 0.975) < 1e-6 and abs(Phi_inv(0.975) - 1.959964) < 1e-5
    # Closed-form gammas vs known values.
    ga, gd, gc = gamma_absmean_gauss(), gamma_drop_gauss(), gamma_clip_gauss()
    assert abs(ga - 0.797885) < 1e-5
    assert gd < ga and gc < ga
    # Closed-form trit error vs numeric integration of the same integrand.
    def numeric(s, n=200000, lim=9.0):
        h = 2 * lim / n
        acc = 0.0
        for i in range(n):
            x = -lim + (i + 0.5) * h
            q = s * max(-1, min(1, round(x / s)))
            acc += (x - q) ** 2 * phi(x) * h
        return acc
    for s in (ga, gd, gc, 1.2):
        assert abs(trit_mse_gauss(s) - numeric(s)) < 1e-6, s
    tab = analytic_table()
    assert abs(tab["absmean"]["rel_l2"] - 0.5132) < 2e-3, tab  # docs: ~0.54 at n=128
    assert tab["optimal"]["rel_l2"] <= min(tab[k]["rel_l2"] for k in GAUSS_GAMMAS) + 1e-9
    # Closed-form gammas vs Monte Carlo on a large Gaussian sample.
    rng = random.Random(0)
    big = [rng.gauss(0.0, 1.0) for _ in range(40000)]
    for k in GAUSS_GAMMAS:
        assert abs(GAMMAS[k](big) - GAUSS_GAMMAS[k]()) < 0.01, k
    # Measured block error tracks the analytic expectation.
    for k in GAUSS_GAMMAS:
        b = block_budget(big[:BLOCK * 64], k)
        assert abs(b["measured"] - b["expected"]) < 0.02, (k, b["measured"], b["expected"])
    # Manifest: 8 Gaussian blocks of 128 admit, chain is deterministic and forward-only.
    w = [rng.gauss(0.0, 0.02) for _ in range(BLOCK * 8)]
    m = i2sd_manifest(w)
    assert m["ok"] and m["admit_all"] and len(m["blocks"]) == 8 and len(m["close"]) == 16
    assert m["close"] == i2sd_manifest(w)["close"]
    assert m["close"] != i2sd_manifest(w, prev=m["close"])["close"]
    assert m["close"] != i2sd_manifest(w, variant="drop")["close"]
    for k in GAUSS_GAMMAS:
        assert i2sd_manifest(w, variant=k)["admit_all"], k
    assert i2sd_manifest(w, variant="nope")["ok"] is False
    return {"selftest": "pass", "analytic": tab, "close": m["close"], "endpoint": I2SD_ENDPOINT,
            "roadmap": ROADMAP_ITEM, "ports": list(PORTS)}


def main(argv) -> int:
    if "--selftest" in argv:
        print(json.dumps(self_check(), sort_keys=True))
        return 0
    rng = random.Random(1)
    w = [rng.gauss(0.0, 0.02) for _ in range(BLOCK * 4)]
    m = i2sd_manifest(w)
    print(json.dumps({"analytic": analytic_table(), "manifest": m}, sort_keys=True))
    return 0 if m.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
