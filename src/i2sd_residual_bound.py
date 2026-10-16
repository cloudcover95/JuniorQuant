"""i2sd residual bound: Winsor P5/P95 per-element and matvec bound on the 2B4T reconstruct.

Stdlib only.

Roadmap item: JuniorHome docs/ROADMAP.md step 3, "i2sd userspace - 127.0.0.1:8767.
No docker.sock." (Quant Chief, pair A, 2026-10-16 10:14 MT). This file is the
quant-error and Winsor-bound side of that item, complementary to the buddy file
JuniorLLM scripts/i2s_2b4t_key_pack.py (Trit Pack & I2_S, e8af871), which packs
4 trits per byte (2B4T) but says nothing about how far the reconstruct is from W.

New capability vs the earlier Winsor files (src/winsor_clip.py, src/i2sd_quant_budget.py,
JuniorStock risk/winsor_quant_error.py, security/winsor_outlier_health.py): a hard,
deterministic (not expected-value) residual certificate for the reconstruct
w_hat = gamma * clip(round(w / gamma), -1, 1):
  - two-sided signed Winsor window [P5(W), P95(W)], M = max(|P5|, |P95|)
  - every in-window element obeys |w - w_hat| <= R, R = max(min(M, gamma/2), M - gamma)
  - out-of-window elements (about 10%) carry their exact residual | |w| - gamma |
  - per 2B4T unit (4 trits = 1 byte) L_inf certificate
  - i2sd matvec bound: |y - y_hat| <= min(R*sum_in|x| + sum_out|x||r|, ||r||_2 ||x||_2)
for any of the three gammas in docs/WINSOR.md (BitNet absmean, Junior drop 1/16, P95 clip).
Bounds are certified per row and closed row by row under the close rule.

Close rule: sha256(prev || pack || text)[:16], pack = 2B4T bytes of the row's trits
(I2_S codes 0->0, 1->1, -1->2, MSB first, zero-padded to 4). Rate 0.05 held.
Compatible engines: BitNet-mlx, JuniorLLM, web3node, stocksnode, JuniorQuant,
JuniorStock, i2sd userspace (same pack and close vector).
Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
The i2sd endpoint is named only (http://127.0.0.1:8767); no socket is opened here.
Bind 127.0.0.1 only. mint/socket/layer_one/fetch/ue5_launch false. No docker.sock.
No model pull, no GGUF download, no network.

Usage: python3 i2sd_residual_bound.py [--selftest]
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
BUDDY_FILE = "JuniorLLM scripts/i2s_2b4t_key_pack.py (e8af871)"
GENESIS = "0" * 16
UNIT = 4  # 2B4T: 4 trits per byte
P_LO, P_HI = 5.0, 95.0
EPS = 1e-8
TOL = 1e-12

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


def pack_2b4t(trits) -> bytes:
    t = list(trits)
    t += [0] * (-len(t) % UNIT)
    return pack(t)


def close_rule(prev: str, pk: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode() + pk + text.encode()).hexdigest()[:16]


# ---- gammas (docs/WINSOR.md) and signed percentiles --------------------------------

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


def elem_bound(m: float, gamma: float) -> float:
    """Sup of |w - gamma*clip(round(w/gamma))| over |w| <= m."""
    return max(min(m, gamma / 2.0), m - gamma)


# ---- per-row residual certificate --------------------------------------------------

def row_certificate(w, variant: str = "absmean") -> dict:
    g = GAMMAS[variant](w)
    lo, hi = percentile(w, P_LO), percentile(w, P_HI)
    m = max(abs(lo), abs(hi))
    r_bound = elem_bound(m, g + EPS)
    t = ternarize(w, g)
    resid = [x - g * q for x, q in zip(w, t)]
    inside = [lo - TOL <= x <= hi + TOL for x in w]
    units = []
    for u in range(0, len(w), UNIT):
        idx = range(u, min(u + UNIT, len(w)))
        cap = max((r_bound if inside[i] else abs(resid[i])) for i in idx)
        units.append(cap)
    return {"gamma": g, "p5": lo, "p95": hi, "m": m, "r_bound": r_bound,
            "trits": t, "resid": resid, "inside": inside, "unit_linf": units,
            "outliers": sum(1 for f in inside if not f),
            "max_in": max((abs(r) for r, f in zip(resid, inside) if f), default=0.0)}


def matvec_bound(cert: dict, x) -> float:
    """Certified |sum x_i w_i - sum x_i w_hat_i| without knowing W beyond the cert."""
    a = sum(abs(xi) * (cert["r_bound"] if f else abs(r))
            for xi, r, f in zip(x, cert["resid"], cert["inside"]))
    l2r = math.sqrt(sum(r * r for r in cert["resid"]))
    l2x = math.sqrt(sum(xi * xi for xi in x))
    return min(a, l2r * l2x)


def i2sd_residual_manifest(rows, x, variant: str = "absmean", prev: str = GENESIS) -> dict:
    """Per-row certified residual and matvec bounds for i2sd, closed row by row."""
    if BIND != "127.0.0.1" or not I2SD_ENDPOINT.startswith("http://127.0.0.1:"):
        return {"ok": False, "why": "bind"}
    if MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH:
        return {"ok": False, "why": "flags"}
    if variant not in GAMMAS:
        return {"ok": False, "why": "variant"}
    out, close = [], prev
    for i, w in enumerate(rows):
        if len(w) != len(x):
            return {"ok": False, "why": "shape", "row": i}
        c = row_certificate(w, variant)
        y = sum(a * b for a, b in zip(x, w))
        yh = sum(a * c["gamma"] * q for a, q in zip(x, c["trits"]))
        mb = matvec_bound(c, x)
        held = c["max_in"] <= c["r_bound"] + 1e-9 and abs(y - yh) <= mb + 1e-9
        close = close_rule(close, pack_2b4t(c["trits"]),
                           "i2sd|resid|%s|%d|%.2f" % (variant, i, RATE))
        out.append({"row": i, "gamma": round(c["gamma"], 6), "r_bound": round(c["r_bound"], 6),
                    "max_in": round(c["max_in"], 6), "outliers": c["outliers"],
                    "err": round(abs(y - yh), 8), "bound": round(mb, 8),
                    "held": held, "close": close})
    return {"ok": True, "endpoint": I2SD_ENDPOINT, "variant": variant, "rate": RATE,
            "window": [P_LO, P_HI], "unit": UNIT, "held_all": all(r["held"] for r in out),
            "rows": out, "close": close}


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
    assert pack_2b4t(t) == p and len(pack_2b4t([1, 0, -1, -1])) == 1
    assert close_rule(GENESIS, p, "web3") == hashlib.sha256(GENESIS.encode() + p + b"web3").hexdigest()[:16]
    # elem_bound is the exact sup: brute force on a fine grid.
    for g in (0.3, 1.0, 2.5):
        for m in (0.1, 0.5, 1.0, 1.6, 3.0, 7.0):
            sup = max(abs(v - g * max(-1, min(1, round(v / g))))
                      for v in (m * k / 4000.0 for k in range(-4000, 4001)))
            assert sup <= elem_bound(m, g) + 1e-9 and sup >= elem_bound(m, g) - 2e-3 * max(m, g), (g, m)
    # Certificates hold on Gaussian, Laplace and spiky rows for every gamma.
    rng = random.Random(0)
    dists = {
        "gauss": lambda: rng.gauss(0.0, 0.02),
        "laplace": lambda: rng.expovariate(50.0) * (1 if rng.random() < 0.5 else -1),
        "spiky": lambda: rng.gauss(0.0, 0.02) * (8.0 if rng.random() < 0.03 else 1.0),
    }
    x = [rng.gauss(0.0, 1.0) for _ in range(256)]
    for name, draw in dists.items():
        rows = [[draw() for _ in range(256)] for _ in range(6)]
        for k in GAMMAS:
            m = i2sd_residual_manifest(rows, x, variant=k)
            assert m["ok"] and m["held_all"], (name, k, m["rows"][0])
            for r in m["rows"]:
                assert r["outliers"] <= int(0.10 * 256) + 2, (name, k, r)
                assert r["err"] <= r["bound"] + 1e-9
    # Per-unit L_inf caps cover every element of their 2B4T unit.
    c = row_certificate([rng.gauss(0.0, 0.02) for _ in range(130)])
    assert len(c["unit_linf"]) == 33
    for u, cap in enumerate(c["unit_linf"]):
        for i in range(u * UNIT, min(u * UNIT + UNIT, 130)):
            assert abs(c["resid"][i]) <= cap + 1e-12
    # Tightness: the certified matvec bound is not vacuous vs ||w||_2 ||x||_2.
    w = [rng.gauss(0.0, 0.02) for _ in range(256)]
    cw = row_certificate(w)
    assert matvec_bound(cw, x) < math.sqrt(sum(v * v for v in w)) * math.sqrt(sum(v * v for v in x))
    # Chain: deterministic, forward-only, variant-sensitive, shape and variant guarded.
    rows = [[rng.gauss(0.0, 0.02) for _ in range(256)] for _ in range(4)]
    a = i2sd_residual_manifest(rows, x)
    assert len(a["close"]) == 16 and a["close"] == i2sd_residual_manifest(rows, x)["close"]
    assert a["close"] != i2sd_residual_manifest(rows, x, prev=a["close"])["close"]
    assert a["close"] != i2sd_residual_manifest(rows, x, variant="drop")["close"]
    assert i2sd_residual_manifest(rows, x[:-1])["why"] == "shape"
    assert i2sd_residual_manifest(rows, x, variant="nope")["ok"] is False
    return {"selftest": "pass", "close": a["close"], "endpoint": I2SD_ENDPOINT,
            "roadmap": ROADMAP_ITEM, "buddy": BUDDY_FILE, "ports": list(PORTS),
            "rate": RATE, "close_rule": "sha256(prev || pack || text)[:16]",
            "r_bound_row0": a["rows"][0]["r_bound"], "err_row0": a["rows"][0]["err"],
            "bound_row0": a["rows"][0]["bound"]}


def main(argv) -> int:
    if "--selftest" in argv:
        print(json.dumps(self_check(), sort_keys=True))
        return 0
    rng = random.Random(1)
    x = [rng.gauss(0.0, 1.0) for _ in range(128)]
    rows = [[rng.gauss(0.0, 0.02) for _ in range(128)] for _ in range(4)]
    m = i2sd_residual_manifest(rows, x)
    print(json.dumps(m, sort_keys=True))
    return 0 if m.get("ok") and m.get("held_all") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
