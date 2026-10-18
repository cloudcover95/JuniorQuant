"""Live beta step 6: cross-engine Winsor gamma reproducibility bound (no-delete packs). Stdlib only.

Roadmap item: JuniorLLM docs/BETA_TO_OS.md "Live beta" step 6 ("No deletes. No
>8GB pull. bitnetd on `127.0.0.1:8765` only"), run end to end on the JuniorOS
overlay host; docs/COMPILED_BACKLOG.md "A - live beta suite". Quant Chief pair A,
slot 2026-10-18 14:20 MT (Verifier's known open gap: step 6). Winsor & Quant Error
side: the quant-error bound behind "no deletes". Trit Pack & I2_S covers the
step-6 guard trits (diff / size manifest / ss capture) in a different file.

The gap: "no deletes" means a ternary pack, once written, is append-only. That is
only safe if every engine that re-derives gamma (bitnetd 127.0.0.1:8765, i2sd
userspace 127.0.0.1:8767, BitNet-mlx, JuniorLLM) gets the same trits; otherwise a
re-pack would have to rewrite (delete) old codes. Engines differ in how they sum
|W| (f32 naive, f32 pairwise, f64 naive, f64 fsum), in the percentile convention
for P95, and in the BitNet divisor eps. Nothing so far bounds how many trits that
can flip, or what it costs in quant error.

New capability, for each Winsor gamma (JuniorHome docs/WINSOR.md):
  absmean  gamma = mean |W|
  drop     gamma = mean |W| after dropping the top 1/16 (Junior drop)
  clip     gamma = mean min(|W|, P95) (P95 clip, reference P95 = linear interp)
and each arithmetic scheme s with unit roundoff u (f32 2^-24, f64 2^-53):
  sum      nonnegative terms, |S_hat - S| <= g_k S, g_k = k u / (1 - k u),
           k = n - 1 (naive), ceil(log2 n) (pairwise), 1 (fsum, correctly rounded)
  mean     e = (1 + g_k)(1 + u)(1 + 2 u_n) - 1, u_n = u when n is not exact
  bracket  [gamma (1 - e), gamma (1 + e)], computed from the exact rational
           gamma (fractions.Fraction); for clip, gamma is monotone in the
           threshold, so the P95 bracket over conventions (linear, lower, higher,
           nearest-rank, +-2u) maps to the gamma bracket at its endpoints
  at-risk  trit code = RoundClip(W / (gamma + eps), -1, 1); a trit can differ
           from the exact reference only if |W| lies in the band
           [(g_lo + eps_lo)/2 (1 - 3u), (g_hi + eps_hi)/2 (1 + 3u)] (division
           or reciprocal multiply rounding, ties at exactly 0.5 included)
  dSSE     per element, max over its possible codes and gamma' in the bracket
           of |(|W| - c gamma')^2 - (|W| - c_ref gamma)^2|, summed, over the
           reference SSE: rel_dsse
Verdict trit per (gamma, scheme): +1 no at-risk trit (bit-identical pack across
engines, append-only safe); 0 at-risk share <= RATE and rel_dsse <= RATE (hold:
pin one engine); -1 otherwise. An extra eps_mix row uses f64 fsum with BitNet
eps in {0, 1e-6, 1e-5}. The selftest also emulates every scheme bit-exactly
(f32 via struct round trips) and checks that the realized gamma lies in its
bracket and every realized flip lies in the at-risk set.

End-to-end exit code (step6 rc): 1 when any row is -1 or the weights file is
over the 8 GB pull limit (checked by size before reading, never read), 0 when
the pinned f64_fsum rows are all +1 and no row is -1, else 2 (hold).

Close rule: sha256(prev || pack || text)[:16], GENESIS prev = "0" * 16. Per row:
pack = I2_S 2-bit code of its verdict (trit + 1, four per byte, MSB first),
text = canonical JSON of the row (floats as float.hex). Rows chain in order,
then the report key: pack = I2_S codes of all verdicts, text = canonical JSON.
Rate 0.05 is both the at-risk trit share and the rel_dsse budget.
Compatible engines: bitnetd 127.0.0.1:8765, i2sd userspace 127.0.0.1:8767 (named,
never dialed), BitNet-mlx, JuniorLLM, JuniorQuant, JuniorOS overlay.
Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
Bind 127.0.0.1 only. mint/socket/layer_one/fetch/ue5_launch false.
No model pull, no GGUF download, no network, no subprocess, no team import.
Weights come only from a local text file of floats (--weights) or the built-in
synthetic selftest tensor (example data, not a real model).

Usage:
  python3 src/live_beta_step6_gamma_repro_bound.py --selftest
  python3 src/live_beta_step6_gamma_repro_bound.py --weights W.txt [--prev HEX16]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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
BITNETD = f"{BIND}:8765"
I2SD = f"{BIND}:8767"
PULL_LIMIT = 8 * 10 ** 9  # "No >8GB pull"
GENESIS = "0" * 16
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
ENGINES = ("bitnetd 127.0.0.1:8765", "i2sd 127.0.0.1:8767", "BitNet-mlx",
           "JuniorLLM", "JuniorQuant", "JuniorOS overlay")
ROADMAP_ITEM = ("JuniorLLM docs/BETA_TO_OS.md Live beta step 6: no deletes, no >8GB "
                "pull, bitnetd on 127.0.0.1:8765 only (COMPILED_BACKLOG A)")
U32 = 2.0 ** -24
U64 = 2.0 ** -53
EPS_REF = 1e-5  # BitNet RoundClip divisor eps
EPS_MIX = (0.0, 1e-6, 1e-5)
P95 = Fraction(95, 100)
GAMMAS = ("absmean", "drop", "clip")
SCHEMES = ("f32_naive", "f32_pairwise", "f64_naive", "f64_fsum")
PINNED = "f64_fsum"


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def unit(scheme: str) -> float:
    return U32 if scheme.startswith("f32") else U64


def g(k: int, u: float) -> float:
    if k * u >= 1.0:
        raise ValueError("tensor too long for this bound")
    return k * u / (1.0 - k * u)


def mean_rel_err(scheme: str, n: int) -> float:
    u = unit(scheme)
    k = {"f32_naive": n - 1, "f64_naive": n - 1, "f32_pairwise": max(1, math.ceil(math.log2(max(n, 2)))),
         "f64_fsum": 1}[scheme]
    u_n = u if n > (2 ** 24 if u == U32 else 2 ** 53) else 0.0
    return (1.0 + g(k, u)) * (1.0 + u) * (1.0 + 2.0 * u_n) - 1.0


# ---- exact reference gammas (rational) ---------------------------------------

def p95_candidates(a_sorted: list[float]) -> tuple[Fraction, list[Fraction]]:
    n = len(a_sorted)
    pos = P95 * (n - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    xl, xh = Fraction(a_sorted[lo]), Fraction(a_sorted[hi])
    linear = xl + (xh - xl) * (pos - lo)
    nr = Fraction(a_sorted[max(0, math.ceil(P95 * n) - 1)])
    return linear, [linear, xl, xh, nr]


def gamma_exact(kind: str, a: list[float], a_sorted: list[float], t: Fraction | None = None) -> Fraction:
    n = len(a)
    if kind == "absmean":
        return sum(map(Fraction, a), Fraction(0)) / n
    if kind == "drop":
        keep = a_sorted[: n - n // 16] or a_sorted
        return sum(map(Fraction, keep), Fraction(0)) / len(keep)
    if kind == "clip":
        if t is None:
            t = p95_candidates(a_sorted)[0]
        return sum((min(Fraction(x), t) for x in a), Fraction(0)) / n
    raise ValueError(kind)


def gamma_bracket(kind: str, scheme: str, a, a_sorted) -> tuple[Fraction, float, float]:
    n = len(a)
    ref = gamma_exact(kind, a, a_sorted)
    m = n - n // 16 if kind == "drop" else n
    e = mean_rel_err(scheme, m) + 4 * U64  # outward slack for float conversion
    if kind == "clip":
        u = unit(scheme)
        _, cands = p95_candidates(a_sorted)
        t_lo = min(cands) * Fraction(1.0 - 2 * u)
        t_hi = max(cands) * Fraction(1.0 + 2 * u)
        lo_c, hi_c = gamma_exact("clip", a, a_sorted, t_lo), gamma_exact("clip", a, a_sorted, t_hi)
        return ref, float(lo_c) * (1.0 - e), float(hi_c) * (1.0 + e)
    return ref, float(ref) * (1.0 - e), float(ref) * (1.0 + e)


# ---- bit-exact emulation of each scheme (selftest evidence) -------------------

def _sum(xs: list[float], scheme: str) -> float:
    if scheme == "f64_fsum":
        return math.fsum(xs)
    if scheme == "f64_naive":
        s = 0.0
        for x in xs:
            s += x
        return s
    if scheme == "f32_naive":
        s = 0.0
        for x in xs:
            s = f32(s + x)
        return s
    if scheme == "f32_pairwise":
        def pw(lo: int, hi: int) -> float:
            if hi - lo == 1:
                return xs[lo]
            mid = (lo + hi) // 2
            return f32(pw(lo, mid) + pw(mid, hi))
        return pw(0, len(xs)) if xs else 0.0
    raise ValueError(scheme)


def gamma_emulated(kind: str, scheme: str, a, a_sorted) -> float:
    rnd = f32 if scheme.startswith("f32") else float
    n = len(a)
    if kind == "absmean":
        return rnd(_sum(a, scheme) / n)
    if kind == "drop":
        keep = a_sorted[: n - n // 16] or a_sorted
        return rnd(_sum(keep, scheme) / len(keep))
    pos = 0.95 * (n - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    xl, xh = a_sorted[lo], a_sorted[hi]
    t = rnd(xl + rnd(rnd(xh - xl) * rnd(pos - lo)))
    return rnd(_sum([min(x, t) for x in a], scheme) / n)


def code_emulated(w: float, div: float, scheme: str) -> int:
    rnd = f32 if scheme.startswith("f32") else float
    r = round(rnd(abs(w) / div))  # half-even, like torch.round
    return int(math.copysign(min(r, 1), w)) if r else 0


# ---- rows ---------------------------------------------------------------------

def bound_row(kind: str, scheme: str, w: list[float], a, a_sorted, eps_set=(EPS_REF,)) -> dict:
    n = len(w)
    u = unit(scheme)
    ref, g_lo, g_hi = gamma_bracket(kind, scheme, a, a_sorted)
    eps_ref = Fraction(EPS_REF)
    div_ref = ref + eps_ref
    band_lo = (g_lo + min(eps_set)) / 2.0 * (1.0 - 3 * u)
    band_hi = (g_hi + max(eps_set)) / 2.0 * (1.0 + 3 * u)
    gf = float(ref)
    at_risk, dsse, sse = [], 0.0, 0.0
    for i, x in enumerate(a):
        c_ref = 1 if 2 * Fraction(x) > div_ref else 0
        e_ref = (x - c_ref * gf) ** 2
        sse += e_ref
        risky = band_lo <= x <= band_hi
        if risky:
            at_risk.append(i)
        codes = (0, 1) if risky else (c_ref,)
        gs = (g_lo, g_hi, min(max(x, g_lo), g_hi))
        dsse += max(abs((x - c * gp) ** 2 - e_ref) for c in codes for gp in gs)
    rel = dsse / sse if sse > 0 else 0.0
    share = len(at_risk) / n
    verdict = 1 if not at_risk else (0 if share <= RATE and rel <= RATE else -1)
    return {"gamma": kind, "scheme": scheme, "eps": list(eps_set), "n": n,
            "gamma_ref": gf, "gamma_lo": g_lo, "gamma_hi": g_hi, "rel_e": (g_hi - g_lo) / (2 * gf) if gf else 0.0,
            "at_risk": len(at_risk), "at_risk_share": share, "rel_dsse": rel, "verdict": verdict,
            "_risk_set": at_risk, "_ref": ref}


# ---- I2_S pack + close rule -------------------------------------------------------

def i2s_pack(trits: list[int]) -> bytes:
    out = bytearray()
    for i in range(0, len(trits), 4):
        b = 0
        chunk = list(trits[i:i + 4]) + [0] * (4 - len(trits[i:i + 4]))
        for t in chunk:
            if t not in (-1, 0, 1):
                raise ValueError("not a trit")
            b = (b << 2) | (t + 1)
        out.append(b)
    return bytes(out)


def i2s_unpack(data: bytes, n: int) -> list[int]:
    out = []
    for b in data:
        for sh in (6, 4, 2, 0):
            c = (b >> sh) & 3
            if c == 3:
                raise ValueError("bad I2_S code")
            out.append(c - 1)
    return out[:n]


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


def pull_guard(size: int) -> int:
    """Step-6 size trit for the local weights file: +1 within 8 GB, -1 over."""
    return 1 if 0 <= size <= PULL_LIMIT else -1


def report(w: list[float], prev: str = GENESIS) -> dict:
    a = [abs(f32(x)) for x in w]  # canonical f32 weights, shared by every engine
    a_sorted = sorted(a)
    rows = [bound_row(k, s, w, a, a_sorted) for k in GAMMAS for s in SCHEMES]
    rows += [bound_row(k, PINNED, w, a, a_sorted, EPS_MIX) | {"scheme": "eps_mix"} for k in GAMMAS]
    key = prev
    for r in rows:
        r["key"] = close_rule(key, i2s_pack([r["verdict"]]), canon(r))
        key = r["key"]
    verdicts = [r["verdict"] for r in rows]
    pinned = [r["verdict"] for r in rows if r["scheme"] == PINNED]
    rc = 1 if -1 in verdicts else (0 if all(v == 1 for v in pinned) else 2)
    cheapest = {}
    for k in GAMMAS:
        ok = [r["scheme"] for r in rows if r["gamma"] == k and r["scheme"] in SCHEMES and r["verdict"] == 1]
        cheapest[k] = ok[0] if ok else None
    rep = {"item": ROADMAP_ITEM, "rate": RATE, "bind": BIND, "bitnetd": BITNETD, "i2sd": I2SD,
           "ports": list(PORTS), "engines": list(ENGINES), "rows": rows,
           "cheapest_bit_stable": cheapest, "step6_rc": rc,
           "flags": {"mint": MINT, "socket": SOCKET, "layer_one": LAYER_ONE, "fetch": FETCH,
                     "ue5_launch": UE5_LAUNCH}}
    rep["key"] = close_rule(key, i2s_pack(verdicts), canon(rep))
    return rep


def synthetic(n: int = 4096, seed: int = 7) -> list[float]:
    """Example data only: Gaussian-like weights (std 0.02) with a few outliers."""
    rng = random.Random(seed)
    w = [rng.gauss(0.0, 0.02) for _ in range(n)]
    for i in range(0, n, 509):
        w[i] *= 8.0
    return w


def selftest() -> None:
    assert BIND == "127.0.0.1" and not any((MINT, SOCKET, LAYER_ONE, FETCH, UE5_LAUNCH))
    t = [1, 0, -1, 1, -1, 0, 0]
    assert i2s_unpack(i2s_pack(t), len(t)) == t
    assert close_rule(GENESIS, b"\x02", "x") == hashlib.sha256(b"0" * 16 + b"\x02x").hexdigest()[:16]
    assert pull_guard(PULL_LIMIT) == 1 and pull_guard(PULL_LIMIT + 1) == -1
    w = synthetic()
    a = [abs(f32(x)) for x in w]
    a_sorted = sorted(a)
    # every realized engine gamma lies in its bracket, every realized flip is at-risk
    for k in GAMMAS:
        widths = []
        for s in SCHEMES:
            r = bound_row(k, s, w, a, a_sorted)
            gh = gamma_emulated(k, s, a, a_sorted)
            assert r["gamma_lo"] <= gh <= r["gamma_hi"], (k, s, gh, r["gamma_lo"], r["gamma_hi"])
            div_ref = r["_ref"] + Fraction(EPS_REF)
            rnd = f32 if s.startswith("f32") else float
            risk = set(r["_risk_set"])
            for i, x in enumerate(a):
                c_ref = 1 if 2 * Fraction(x) > div_ref else 0
                if code_emulated(x, rnd(gh + EPS_REF), s) != c_ref:
                    assert i in risk, (k, s, i)
            widths.append((r["gamma_hi"] - r["gamma_lo"], r["at_risk"]))
        # f32 naive bracket is the widest; f64 fsum the narrowest
        assert widths[0][0] >= widths[-1][0] and widths[0][1] >= widths[-1][1]
    # a planted near-tie at the 0.5 boundary (fixed point x = (mean + eps)/2,
    # off by one f32 rounding) is at-risk for f32 engines, so never +1 there
    w2 = [0.01, -0.03, 0.02, 0.05, -0.04, 0.0, 0.015, 0.025]
    a2 = [abs(f32(x)) for x in w2]
    s2, n2 = sum(map(Fraction, a2)), len(a2)
    w2.append(f32(float((s2 + Fraction(EPS_REF) * (n2 + 1)) / (2 * n2 + 1))))
    a2 = [abs(f32(x)) for x in w2]
    r2 = bound_row("absmean", "f32_naive", w2, a2, sorted(a2))
    assert (len(a2) - 1) in r2["_risk_set"] and r2["verdict"] != 1
    # end-to-end report: deterministic key, rc in {0, 1, 2}, chain uses the close rule
    rep1, rep2 = report(w), report(w)
    assert rep1["key"] == rep2["key"] and rep1["step6_rc"] in (0, 1, 2)
    assert report(w, prev="f" * 16)["key"] != rep1["key"]
    print(json.dumps({"selftest": "pass", "key": rep1["key"], "step6_rc": rep1["step6_rc"],
                      "cheapest_bit_stable": rep1["cheapest_bit_stable"],
                      "rows": [(r["gamma"], r["scheme"], r["at_risk"], round(r["rel_dsse"], 8), r["verdict"])
                               for r in rep1["rows"]]}))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--weights", help="local text file of floats (whitespace separated)")
    ap.add_argument("--prev", default=GENESIS)
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return 0
    if args.weights:
        size = os.path.getsize(args.weights)
        if pull_guard(size) != 1:
            print(json.dumps({"step6_rc": 1, "reason": "weights file over 8 GB pull limit", "bytes": size}))
            return 1
        with open(args.weights, encoding="utf-8") as fh:
            w = [float(tok) for line in fh for tok in line.split()]
    else:
        w = synthetic()
    if len(args.prev) != 16 or any(c not in "0123456789abcdef" for c in args.prev):
        raise SystemExit("--prev must be 16 hex chars")
    rep = report(w, args.prev)
    print(canon(rep))
    return rep["step6_rc"]


if __name__ == "__main__":
    sys.exit(main())
