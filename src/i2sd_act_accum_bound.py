"""i2sd activation x ternary weight joint error bound and integer accumulator headroom.

Stdlib only.

Roadmap item: JuniorHome docs/ROADMAP.md step 3, "i2sd userspace - 127.0.0.1:8767.
No docker.sock." (Quant Chief, pair A, 2026-10-16 18:12 MT, third naming). This is
the quant-error and Winsor-bound side of that item. The buddy file for this cycle is
BitNet-mlx scripts/i2sd_userspace_policy_key.py (Trit Pack & I2_S, 962f8e3), a launch
policy key; this file is a different repo and path and does not wrap it.

New angle vs the earlier i2sd files (src/i2sd_quant_budget.py was an expected-value
weight-only budget; src/i2sd_residual_bound.py was a weight-only P5/P95 residual
certificate): both of those treat the activation x as exact float. i2sd actually runs
an int8 activation times a ternary weight with an integer accumulator, as in BitNet
b1.58. This file certifies that real kernel:
  y      = x . w                          (float reference)
  y_int  = sum_i q_i * t_i                (q = int8 activation code, t = trit)
  y_hat  = gamma * s * y_int              (s = activation scale)
  |y - y_hat| <= sum_i |x_i| |w_i - gamma t_i|  +  gamma * sum_{t_i != 0} |x_i - s q_i|
and the a-priori (data-light) form gamma * s/2 * nnz(t) for the activation term.

Activation Winsor variants (new: a Winsor gamma on the activation side):
  absmax : s = max|x| / 127                     (BitNet b1.58 per-token absmax)
  p99    : s = P99(|x|) / 127, values beyond clip to +-127 and carry |x| - 127 s
  p999   : s = P99.9(|x|) / 127, same rule
Weight gammas from docs/WINSOR.md: BitNet absmean mean|W|, Junior drop (mean |W|
after dropping the top 1/16), P95 clip.

Accumulator headroom: |y_int| <= 127 * nnz(t) <= 127 * K, so an int16 lane is exact
iff nnz <= 258 and an int32 lane iff K <= 16909320. The file reports the minimum
safe accumulator width per row and the longest K each width can carry.

Close rule: sha256(prev || pack || text)[:16], pack = I2_S 2-bit codes of the row's
trits (0->0, 1->1, -1->2, MSB first, zero-padded), text names variant, row, rate.
Rate 0.05 held. Compatible engines: BitNet-mlx, JuniorLLM, web3node, stocksnode,
JuniorQuant, JuniorStock, i2sd userspace (same pack and close vector).
Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
The i2sd endpoint is named only (http://127.0.0.1:8767); no socket is opened here.
Bind 127.0.0.1 only. mint/socket/layer_one/fetch/ue5_launch false. No docker.sock.
No model pull, no GGUF download, no network.

Usage: python3 i2sd_act_accum_bound.py [--selftest]
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
BUDDY_FILE = "BitNet-mlx scripts/i2sd_userspace_policy_key.py (962f8e3)"
GENESIS = "0" * 16
QMAX = 127
EPS = 1e-8
INT16_MAX = 2 ** 15 - 1
INT32_MAX = 2 ** 31 - 1

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


# ---- weight gammas (docs/WINSOR.md) -----------------------------------------------

def percentile(xs, q: float) -> float:
    s = sorted(xs)
    k = (len(s) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def gamma_absmean(w) -> float:
    return sum(abs(v) for v in w) / len(w)


def gamma_drop(w) -> float:
    a = sorted(abs(v) for v in w)
    keep = a[: len(a) - len(a) // 16] or a
    return sum(keep) / len(keep)


def gamma_clip(w, q: float = 95.0) -> float:
    a = [abs(v) for v in w]
    p = percentile(a, q)
    return sum(min(v, p) for v in a) / len(a)


W_GAMMAS = {"absmean": gamma_absmean, "drop": gamma_drop, "clip": gamma_clip}


def ternarize(w, gamma: float) -> list:
    g = gamma + EPS
    return [max(-1, min(1, round(v / g))) for v in w]


# ---- activation Winsor scales and int8 codes --------------------------------------

def act_scale(x, variant: str) -> float:
    a = [abs(v) for v in x]
    if variant == "absmax":
        m = max(a)
    elif variant == "p99":
        m = percentile(a, 99.0)
    elif variant == "p999":
        m = percentile(a, 99.9)
    else:
        raise KeyError(variant)
    return max(m, EPS) / QMAX


A_VARIANTS = ("absmax", "p99", "p999")


def quant_act(x, s: float) -> list:
    return [max(-QMAX, min(QMAX, round(v / s))) for v in x]


def act_elem_cap(v: float, s: float) -> float:
    """Certified |v - s*q(v)|: s/2 inside the clip, |v| - 127 s when clipped."""
    return max(s / 2.0, abs(v) - QMAX * s)


# ---- accumulator headroom ----------------------------------------------------------

def accum_width(nnz: int) -> int:
    """Smallest signed lane (bits) that holds 127 * nnz exactly."""
    peak = QMAX * nnz
    for bits in (8, 16, 32, 64):
        if peak <= 2 ** (bits - 1) - 1:
            return bits
    return 128


def max_k(bits: int) -> int:
    return (2 ** (bits - 1) - 1) // QMAX


# ---- per-row joint certificate -----------------------------------------------------

def row_joint(w, x, w_variant: str, a_variant: str) -> dict:
    g = W_GAMMAS[w_variant](w)
    t = ternarize(w, g)
    s = act_scale(x, a_variant)
    q = quant_act(x, s)
    y = sum(a * b for a, b in zip(x, w))
    y_int = sum(qi * ti for qi, ti in zip(q, t))
    y_hat = g * s * y_int
    w_term = sum(abs(xi) * abs(wi - g * ti) for xi, wi, ti in zip(x, w, t))
    a_term = g * sum(act_elem_cap(xi, s) for xi, ti in zip(x, t) if ti)
    nnz = sum(1 for ti in t if ti)
    clipped = sum(1 for xi in x if abs(xi) > QMAX * s + 1e-12)
    a_prior = g * (s / 2.0) * nnz if clipped == 0 else None
    return {"gamma": g, "s": s, "trits": t, "codes": q, "y": y, "y_int": y_int,
            "y_hat": y_hat, "err": abs(y - y_hat), "bound": w_term + a_term,
            "w_term": w_term, "a_term": a_term, "a_prior": a_prior, "nnz": nnz,
            "clipped": clipped, "width": accum_width(nnz),
            "int_ok": abs(y_int) <= QMAX * nnz}


def i2sd_joint_manifest(rows, x, w_variant: str = "absmean", a_variant: str = "absmax",
                        prev: str = GENESIS) -> dict:
    """Per-row W x A error certificate and accumulator width for i2sd, closed row by row."""
    if BIND != "127.0.0.1" or not I2SD_ENDPOINT.startswith("http://127.0.0.1:"):
        return {"ok": False, "why": "bind"}
    if MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH:
        return {"ok": False, "why": "flags"}
    if w_variant not in W_GAMMAS or a_variant not in A_VARIANTS:
        return {"ok": False, "why": "variant"}
    out, close = [], prev
    for i, w in enumerate(rows):
        if len(w) != len(x):
            return {"ok": False, "why": "shape", "row": i}
        c = row_joint(w, x, w_variant, a_variant)
        held = c["err"] <= c["bound"] + 1e-9 and c["int_ok"]
        close = close_rule(close, pack(c["trits"]),
                           "i2sd|joint|%s|%s|%d|%.2f" % (w_variant, a_variant, i, RATE))
        out.append({"row": i, "gamma": round(c["gamma"], 6), "s": round(c["s"], 8),
                    "err": round(c["err"], 8), "bound": round(c["bound"], 8),
                    "w_term": round(c["w_term"], 8), "a_term": round(c["a_term"], 8),
                    "nnz": c["nnz"], "clipped": c["clipped"], "acc_bits": c["width"],
                    "held": held, "close": close})
    return {"ok": True, "endpoint": I2SD_ENDPOINT, "w_variant": w_variant,
            "a_variant": a_variant, "rate": RATE, "k": len(x),
            "acc_bits": max(r["acc_bits"] for r in out) if out else 8,
            "max_k": {"int16": max_k(16), "int32": max_k(32)},
            "held_all": all(r["held"] for r in out), "rows": out, "close": close}


def best_act_variant(rows, x, w_variant: str = "absmean") -> dict:
    """Pick the activation Winsor scale with the smallest total certified bound."""
    tot = {}
    for a in A_VARIANTS:
        m = i2sd_joint_manifest(rows, x, w_variant, a)
        tot[a] = sum(r["bound"] for r in m["rows"])
    pick = min(A_VARIANTS, key=lambda a: (tot[a], A_VARIANTS.index(a)))
    return {"pick": pick, "totals": {k: round(v, 8) for k, v in tot.items()}}


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
    # Accumulator widths: int16 holds exactly 258 nonzero trits, int32 holds 16909320.
    assert max_k(16) == 258 and max_k(32) == 16909320
    assert accum_width(1) == 8 and accum_width(2) == 16 and accum_width(258) == 16
    assert accum_width(259) == 32 and accum_width(16909320) == 32 and accum_width(16909321) == 64
    # act_elem_cap is a true cap: brute force over a grid incl. clipped values.
    for s in (0.01, 0.3, 1.0):
        for k in range(-60000, 60001, 7):
            v = k * s / 300.0
            qv = max(-QMAX, min(QMAX, round(v / s)))
            assert abs(v - s * qv) <= act_elem_cap(v, s) + 1e-12, (s, v)
    # Joint certificate holds for every weight gamma x activation variant x distribution.
    rng = random.Random(0)
    dists = {
        "gauss": lambda: rng.gauss(0.0, 0.02),
        "laplace": lambda: rng.expovariate(50.0) * (1 if rng.random() < 0.5 else -1),
        "spiky": lambda: rng.gauss(0.0, 0.02) * (8.0 if rng.random() < 0.03 else 1.0),
    }
    x = [rng.gauss(0.0, 1.0) * (6.0 if rng.random() < 0.01 else 1.0) for _ in range(512)]
    for name, draw in dists.items():
        rows = [[draw() for _ in range(512)] for _ in range(4)]
        for wv in W_GAMMAS:
            for av in A_VARIANTS:
                m = i2sd_joint_manifest(rows, x, wv, av)
                assert m["ok"] and m["held_all"], (name, wv, av, m["rows"][0])
                assert m["acc_bits"] in (16, 32)
                if av == "absmax":
                    assert all(r["clipped"] == 0 for r in m["rows"])
    # Absmax: the data-light activation term caps the measured one.
    w = [rng.gauss(0.0, 0.02) for _ in range(512)]
    c = row_joint(w, x, "absmean", "absmax")
    assert c["a_prior"] is not None and c["a_term"] <= c["a_prior"] + 1e-12
    # Integer path is exact: y_int equals the integer dot of codes and trits.
    assert c["y_int"] == sum(a * b for a, b in zip(c["codes"], c["trits"]))
    assert all(-QMAX <= qv <= QMAX for qv in c["codes"])
    # Tightness: the certificate is far below Cauchy-Schwarz ||w||_2 ||x||_2.
    cs = math.sqrt(sum(v * v for v in w)) * math.sqrt(sum(v * v for v in x))
    assert c["bound"] < cs
    # Best activation variant is one of the three and is deterministic.
    rows = [[rng.gauss(0.0, 0.02) for _ in range(512)] for _ in range(3)]
    b1, b2 = best_act_variant(rows, x), best_act_variant(rows, x)
    assert b1 == b2 and b1["pick"] in A_VARIANTS
    # Chain: deterministic, forward-only, variant-sensitive, shape and variant guarded.
    a = i2sd_joint_manifest(rows, x)
    assert len(a["close"]) == 16 and a["close"] == i2sd_joint_manifest(rows, x)["close"]
    assert a["close"] != i2sd_joint_manifest(rows, x, prev=a["close"])["close"]
    assert a["close"] != i2sd_joint_manifest(rows, x, "drop")["close"]
    assert a["close"] != i2sd_joint_manifest(rows, x, a_variant="p99")["close"]
    assert i2sd_joint_manifest(rows, x[:-1])["why"] == "shape"
    assert i2sd_joint_manifest(rows, x, a_variant="nope")["ok"] is False
    return {"selftest": "pass", "close": a["close"], "endpoint": I2SD_ENDPOINT,
            "roadmap": ROADMAP_ITEM, "buddy": BUDDY_FILE, "ports": list(PORTS),
            "rate": RATE, "close_rule": "sha256(prev || pack || text)[:16]",
            "acc_bits": a["acc_bits"], "max_k": a["max_k"], "best_act": b1,
            "err_row0": a["rows"][0]["err"], "bound_row0": a["rows"][0]["bound"]}


def main(argv) -> int:
    if "--selftest" in argv:
        print(json.dumps(self_check(), sort_keys=True))
        return 0
    rng = random.Random(1)
    x = [rng.gauss(0.0, 1.0) for _ in range(256)]
    rows = [[rng.gauss(0.0, 0.02) for _ in range(256)] for _ in range(4)]
    m = i2sd_joint_manifest(rows, x)
    print(json.dumps(m, sort_keys=True))
    return 0 if m.get("ok") and m.get("held_all") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
