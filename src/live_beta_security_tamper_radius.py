"""Live beta juniorctl security: certified weight-tamper radius per Winsor gamma. Stdlib only.

Roadmap item: JuniorLLM docs/BETA_TO_OS.md "Live beta", juniorctl security gaps
(docs/COMPILED_BACKLOG.md "A - live beta suite"). Quant Chief pair A, slot
2026-10-19 02:20 MT. Winsor & Quant Error side: the quant-error bound a
`juniorctl security` line can cite for the ternary packs that i2sd userspace
(127.0.0.1:8767) and BitNet ternary inference rely on. Trit Pack & I2_S covers
the I2_S pack side in a different file.

The gap: the security check commits to the I2_S pack hash. Nothing so far says
which weight tampering that hash can and cannot see, or what an unseen tamper
costs. Earlier work (JuniorStock security/winsor_outlier_health.py) counts how
many *outliers* (an L0 attack) break gamma; this file is the dense L-infinity
case: every weight nudged by at most r. It does not wrap or re-derive that file.

New capability, for each Winsor gamma (JuniorHome docs/WINSOR.md):
  absmean  gamma = mean |W|
  drop     gamma = mean |W| after dropping the top 1/16 (Junior drop)
  clip     gamma = mean min(|W|, P95) (P95 clip, P95 = linear interpolation)
  Lemma    each gamma is 1-Lipschitz in L-inf: if |d_i| <= r for all i then
           |gamma(W + d) - gamma(W)| <= r (|.|, order statistics, min and
           means are all 1-Lipschitz).
  code     c_i = RoundClip(W_i / (gamma + eps), -1, 1), half-even, so the
           magnitude bit is 1 iff |W_i| > (gamma + eps) / 2.
  margin   m_i = | |W_i| - (gamma + eps) / 2 |, exact (fractions.Fraction).
  radius   rho_i = (2/3) m_i: |W_i| moves at most r, the threshold at most r/2,
           so any tamper with ||d||_inf < rho_i leaves trit i (and its sign)
           unchanged. Global radius r_star = min_i rho_i: below it the pack
           hash is identical, so the tamper is invisible AND harmless to the
           trits; its only effect is gamma drift <= r (reported as r/gamma).
  stealth  r_rate = the (floor(RATE n) + 1)-th smallest rho_i: no L-inf tamper
           smaller than r_rate can flip more than floor(RATE n) trits.
  budget   at r_sec = RATE * gamma: at-risk count (rho_i <= r_sec) and a
           certified rel dSSE bound, per element the max over its possible
           codes of (|W'| - c gamma')^2 with |W'| in |W| +- r, gamma' in
           gamma +- r, minus the reference error, summed, over reference SSE.
Verdict trit per gamma: +1 at-risk share at r_sec <= RATE (a 5%-of-gamma
tamper can flip at most 5% of trits); 0 share <= 2 RATE; -1 otherwise.
The selftest checks the lemma and the radius with exact rational arithmetic
under random and adversarial (inward/outward) tampers, and shows the bound is
tight for absmean with an explicit attack that flips the weakest trit just
above its radius.

Close rule: sha256(prev || pack || text)[:16], GENESIS prev = "0" * 16. Per row:
pack = I2_S 2-bit code of its verdict (trit + 1, four per byte, MSB first),
text = canonical JSON of the row (floats as float.hex). Rows chain in order,
then the report key: pack = I2_S codes of all verdicts, text = canonical JSON.
Rate 0.05 is the tamper budget (fraction of gamma) and the at-risk trit share.
Compatible engines: i2sd userspace 127.0.0.1:8767, bitnetd 127.0.0.1:8765 (named,
never dialed), BitNet-mlx, JuniorLLM, JuniorQuant, JuniorOS overlay.
Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
Bind 127.0.0.1 only. mint/socket/layer_one/fetch/ue5_launch false.
No model pull, no GGUF download, no network, no subprocess, no team import.
Weights come only from a local text file of floats (--weights) or the built-in
synthetic selftest tensor (example data, not a real model).

Usage:
  python3 src/live_beta_security_tamper_radius.py --selftest
  python3 src/live_beta_security_tamper_radius.py --weights W.txt [--prev HEX16]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import struct
import sys
from fractions import Fraction

RATE = 0.05
MINT = False
SOCKET = False
LAYER_ONE = False
FETCH = False
UE5_LAUNCH = False
BIND = "127.0.0.1"
I2SD = f"{BIND}:8767"
BITNETD = f"{BIND}:8765"
GENESIS = "0" * 16
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
ENGINES = ("i2sd 127.0.0.1:8767", "bitnetd 127.0.0.1:8765", "BitNet-mlx",
           "JuniorLLM", "JuniorQuant", "JuniorOS overlay")
ROADMAP_ITEM = ("JuniorLLM docs/BETA_TO_OS.md Live beta: juniorctl security "
                "(COMPILED_BACKLOG A - live beta suite)")
EPS = Fraction(1e-5)  # BitNet RoundClip divisor eps
P95 = Fraction(95, 100)
GAMMAS = ("absmean", "drop", "clip")
RATE_F = Fraction(1, 20)
assert float(RATE_F) == RATE


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


# ---- exact gammas -------------------------------------------------------------

def gamma(kind: str, a: list[Fraction]) -> Fraction:
    """Exact rational Winsor gamma of magnitudes a (a_i >= 0)."""
    n = len(a)
    if kind == "absmean":
        return sum(a, Fraction(0)) / n
    s = sorted(a)
    if kind == "drop":
        keep = s[: n - n // 16] or s
        return sum(keep, Fraction(0)) / len(keep)
    if kind == "clip":
        pos = P95 * (n - 1)
        lo, hi = math.floor(pos), math.ceil(pos)
        t = s[lo] + (s[hi] - s[lo]) * (pos - lo)
        return sum((min(x, t) for x in a), Fraction(0)) / n
    raise ValueError(kind)


def codes(w: list[Fraction], g: Fraction) -> list[int]:
    half = (g + EPS) / 2
    return [(1 if x > 0 else -1) if abs(x) > half else 0 for x in w]


# ---- certificate --------------------------------------------------------------

def radii(a: list[Fraction], g: Fraction) -> list[Fraction]:
    half = (g + EPS) / 2
    return [abs(x - half) * Fraction(2, 3) for x in a]


def dsse_bound(a: list[Fraction], g: Fraction, rho: list[Fraction], r: Fraction) -> tuple[Fraction, Fraction]:
    half = (g + EPS) / 2
    sse = Fraction(0)
    worst = Fraction(0)
    for x, ri in zip(a, rho):
        c_ref = 1 if x > half else 0
        e_ref = (x - c_ref * g) ** 2
        sse += e_ref
        cs = (0, 1) if ri <= r else (c_ref,)
        e_max = Fraction(0)
        for c in cs:
            if c == 0:
                e = (x + r) ** 2
            else:
                e = (abs(x - g) + 2 * r) ** 2
            e_max = max(e_max, e)
        worst += e_max - e_ref
    return worst, sse


def row(kind: str, a: list[Fraction]) -> dict:
    n = len(a)
    g = gamma(kind, a)
    rho = radii(a, g)
    srt = sorted(rho)
    k_rate = math.floor(RATE_F * n)
    r_star = srt[0]
    r_rate = srt[k_rate] if k_rate < n else srt[-1]
    r_sec = RATE_F * g
    at_risk = sum(1 for ri in rho if ri <= r_sec)
    share = Fraction(at_risk, n)
    worst, sse = dsse_bound(a, g, rho, r_sec)
    rel = worst / sse if sse else Fraction(0)
    verdict = 1 if share <= RATE_F else (0 if share <= 2 * RATE_F else -1)
    return {"gamma": kind, "n": n, "gamma_ref": float(g), "eps": float(EPS),
            "r_star": float(r_star), "r_star_rel": float(r_star / g) if g else 0.0,
            "r_rate": float(r_rate), "r_rate_rel": float(r_rate / g) if g else 0.0,
            "r_sec": float(r_sec), "at_risk": at_risk, "at_risk_share": float(share),
            "rel_dsse_bound": float(rel), "gamma_drift_rel_at_r_sec": RATE,
            "verdict": verdict, "_g": g, "_rho": rho}


# ---- I2_S pack + close rule ---------------------------------------------------

def i2s_pack(trits: list[int]) -> bytes:
    out = bytearray()
    for i in range(0, len(trits), 4):
        chunk = list(trits[i:i + 4]) + [0] * (4 - len(trits[i:i + 4]))
        b = 0
        for t in chunk:
            if t not in (-1, 0, 1):
                raise ValueError("not a trit")
            b = (b << 2) | (t + 1)
        out.append(b)
    return bytes(out)


def close_rule(prev: str, pack: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode() + pack + text.encode()).hexdigest()[:16]


def canon(obj) -> str:
    def fx(v):
        if isinstance(v, float):
            return v.hex()
        if isinstance(v, dict):
            return {k: fx(x) for k, x in v.items() if not k.startswith("_")}
        if isinstance(v, list):
            return [fx(x) for x in v]
        return v
    return json.dumps(fx(obj), sort_keys=True, separators=(",", ":"))


def to_mags(w: list[float]) -> tuple[list[Fraction], list[Fraction]]:
    wf = [Fraction(f32(x)) for x in w]  # canonical f32 weights shared by engines
    return wf, [abs(x) for x in wf]


def report(w: list[float], prev: str = GENESIS) -> dict:
    _, a = to_mags(w)
    rows = [row(k, a) for k in GAMMAS]
    key = prev
    for r in rows:
        r["key"] = close_rule(key, i2s_pack([r["verdict"]]), canon(r))
        key = r["key"]
    verdicts = [r["verdict"] for r in rows]
    best = max(rows, key=lambda r: (r["r_rate_rel"], -GAMMAS.index(r["gamma"])))["gamma"]
    rep = {"check": "juniorctl security: weight-tamper radius", "item": ROADMAP_ITEM,
           "rate": RATE, "bind": BIND, "i2sd": I2SD, "bitnetd": BITNETD,
           "ports": list(PORTS), "engines": list(ENGINES), "rows": rows,
           "most_tamper_robust_gamma": best,
           "flags": {"mint": MINT, "socket": SOCKET, "layer_one": LAYER_ONE, "fetch": FETCH,
                     "ue5_launch": UE5_LAUNCH}}
    rep["key"] = close_rule(key, i2s_pack(verdicts), canon(rep))
    return rep


def synthetic(n: int = 1024, seed: int = 11) -> list[float]:
    """Example data only: Gaussian-like weights (std 0.02) with a few outliers."""
    rng = random.Random(seed)
    w = [rng.gauss(0.0, 0.02) for _ in range(n)]
    for i in range(0, n, 127):
        w[i] *= 8.0
    return w


# ---- selftest -----------------------------------------------------------------

def _tamper(w: list[Fraction], d: list[Fraction]) -> tuple[list[Fraction], list[Fraction]]:
    w2 = [x + y for x, y in zip(w, d)]
    return w2, [abs(x) for x in w2]


def selftest() -> None:
    assert BIND == "127.0.0.1" and not any((MINT, SOCKET, LAYER_ONE, FETCH, UE5_LAUNCH))
    assert i2s_pack([1, 0, -1, 1]) == bytes([0b10010010])
    assert close_rule(GENESIS, b"\x02", "x") == hashlib.sha256(b"0" * 16 + b"\x02x").hexdigest()[:16]
    rng = random.Random(5)
    w = synthetic(512, 3)
    wf, a = to_mags(w)
    rep_rows = {k: row(k, a) for k in GAMMAS}
    for k in GAMMAS:
        r = rep_rows[k]
        g, rho = r["_g"], r["_rho"]
        c0 = codes(wf, g)
        r_star = min(rho)
        # tampers: random signs, all-outward, all-inward, mixed toward each threshold
        scale = Fraction(999, 1000)
        for trial in range(6):
            rad = [r_star * scale, Fraction(r["r_rate"]) * scale, Fraction(r["r_sec"])][trial % 3]
            if trial < 3:
                d = [rad * Fraction(rng.choice((-1, 1))) for _ in wf]
            elif trial == 3:
                d = [rad if x >= 0 else -rad for x in wf]
            elif trial == 4:
                d = [(-rad if x > 0 else rad) if abs(x) >= rad else -x for x in wf]
            else:
                d = [rad * Fraction(rng.randint(-1000, 1000), 1000) for _ in wf]
            w2, a2 = _tamper(wf, d)
            g2 = gamma(k, a2)
            assert abs(g2 - g) <= rad, (k, trial)  # 1-Lipschitz lemma
            c2 = codes(w2, g2)
            for i, (x, y) in enumerate(zip(c0, c2)):
                if x != y:
                    assert rho[i] <= rad, (k, trial, i)  # only at-risk trits flip
            flips = sum(1 for x, y in zip(c0, c2) if x != y)
            if trial % 3 == 0:
                assert flips == 0, (k, trial, flips)  # below r_star the pack is identical
            if trial % 3 == 1:
                assert flips <= math.floor(RATE_F * len(wf)), (k, trial, flips)
    # tightness for absmean: attack the weakest trit just above its radius
    g = rep_rows["absmean"]["_g"]
    rho = rep_rows["absmean"]["_rho"]
    i = min(range(len(rho)), key=lambda j: rho[j])
    half = (g + EPS) / 2
    rad = rho[i] * Fraction(103, 100)
    up = a[i] <= half  # code 0: push it outward, everyone else inward (gamma drops)
    d = []
    for j, x in enumerate(wf):
        s = 1 if x >= 0 else -1
        if j == i:
            d.append(s * rad if up else -s * rad)
        else:
            d.append((-s * min(rad, abs(x))) if up else s * rad)
    w2, a2 = _tamper(wf, d)
    c0, c2 = codes(wf, g), codes(w2, gamma("absmean", a2))
    assert c0[i] != c2[i], "absmean radius should be tight"
    # report determinism and chaining
    rep1, rep2 = report(w), report(w)
    assert rep1["key"] == rep2["key"] and report(w, "f" * 16)["key"] != rep1["key"]
    full = report(synthetic())
    print(json.dumps({"selftest": "pass", "key": full["key"],
                      "most_tamper_robust_gamma": full["most_tamper_robust_gamma"],
                      "rows": [(r["gamma"], f"{r['r_star_rel']:.3e}", round(r["r_rate_rel"], 6),
                                r["at_risk"], round(r["at_risk_share"], 4), round(r["rel_dsse_bound"], 4),
                                r["verdict"]) for r in full["rows"]]}))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--weights", help="local text file of floats (whitespace separated)")
    ap.add_argument("--prev", default=GENESIS)
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return 0
    if len(args.prev) != 16 or any(c not in "0123456789abcdef" for c in args.prev):
        raise SystemExit("--prev must be 16 hex chars")
    if args.weights:
        with open(args.weights, encoding="utf-8") as fh:
            w = [float(tok) for line in fh for tok in line.split()]
    else:
        w = synthetic()
    print(canon(report(w, args.prev)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
