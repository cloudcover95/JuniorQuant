"""Certified error budget behind the UE5 watts>=80 gate (offline, stdlib only).

Roadmap item: JuniorHome docs/ROADMAP.md step 5, "UE5 launched - JUNIOR_UE5=1 and
watts>=80. Default FrameForge2D. os_route.launch false until then." (Quant Chief,
pair A, 2026-10-17 10:11 MT). This is the quant-error and Winsor-bound side of that
item only. It never launches, locates or touches UE5: UE5_LAUNCH stays False and
the output is advisory. The buddy file for this slot is BitNet-mlx
scripts/ue5_launch_gate_i2s_key.py (Trit Pack & I2_S, 219582e), which turns each
reading into a fixed +-5% hysteresis trit. This file is a different repo and path
and does not wrap, test or re-derive it.

New capability vs my earlier files (all of which bounded weight or activation
error inside i2sd/BitNet matvecs): an error budget on the *watt readings* that feed
the gate, and a gate verdict that is certified rather than banded. Per the docs,
watts are a class value handed in (measured false); nothing here reads a sensor.

Error budget for a window of n readings r_i, each quantized with step q (LSB):
  1. quantization  : true value v_i in [r_i - q/2, r_i + q/2]; clip(.) is
                     1-Lipschitz, so the clipped mean moves by at most q/2.
  2. Winsor clip   : limits [L, U] come from the first half of the window
                     (sample split), so the second half's clipped values are
                     bounded in [L, U] and independent of the limits.
  3. sampling      : Hoeffding at delta = RATE = 0.05 on the second half (m reads):
                     h = (U - L) * sqrt(ln(2/delta) / (2 m)).
  LCB = mean_clip - h - q/2,  UCB = mean_clip + h + q/2.
Verdict trit: +1 if LCB >= 80 (certified above), -1 if UCB < 80 (certified
below), 0 otherwise (undecided -> stay on FrameForge2D).
Sample plan: smallest window n = 2m with h + q/2 <= RATE * 80 = 4 W, i.e.
  m >= ln(2/delta) (U - L)^2 / (2 (4 - q/2)^2),
which says when a fixed 5% band is actually wide enough for the noise seen.

Winsor variants for the clip limits (docs/WINSOR.md, applied to the watt series):
  p95   : [P5, P95]                      (two-sided Gemini/P95 clip)
  drop  : drop top and bottom 1/16, clip at the surviving min/max (Junior drop)
  none  : [min, max] of the first half   (raw, shows spike sensitivity)

Close rule: sha256(prev || pack || text)[:16], pack = I2_S 2-bit codes of the
per-variant verdict trits (0->0, 1->1, -1->2, MSB first, zero-padded), text =
canonical JSON of the budget. Rate 0.05 held (delta and band).
Compatible engines: BitNet-mlx, JuniorLLM, web3node, stocksnode, JuniorQuant,
JuniorStock, i2sd userspace, JuniorOS overlay (same pack and close vector).
Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM (named only; no
socket is opened). Bind 127.0.0.1 only. mint/socket/layer_one/fetch/ue5_launch
false. No model pull, no GGUF download, no network, no hardware calls.

Usage: python3 ue5_watts_gate_error_bound.py [--selftest]
       [--watts 81,85,90,...] [--lsb 1.0] [--env-ue5 1] [--prev HEX16]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys

RATE = 0.05
DELTA = RATE
MINT = False
SOCKET = False
LAYER_ONE = False
FETCH = False
UE5_LAUNCH = False
MEASURED = False
BIND = "127.0.0.1"
WATTS_MIN = 80.0
BAND_W = RATE * WATTS_MIN  # 4.0 W
DEFAULT_ROUTE = "FrameForge2D"
GATED_ROUTE = "UE5"
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
ROADMAP_ITEM = "JuniorHome docs/ROADMAP.md #5 UE5 launched (JUNIOR_UE5=1, watts>=80)"
BUDDY_FILE = "BitNet-mlx scripts/ue5_launch_gate_i2s_key.py (219582e)"
ENGINES = ("BitNet-mlx", "JuniorLLM", "web3node", "stocksnode", "JuniorQuant",
           "JuniorStock", "i2sd userspace", "JuniorOS overlay")
GENESIS = "0" * 16
VARIANTS = ("p95", "drop", "none")

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


# ---- Winsor limits on the watt series ---------------------------------------------

def percentile(xs, q: float) -> float:
    s = sorted(xs)
    k = (len(s) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def winsor_limits(xs, variant: str):
    if not xs:
        raise ValueError("need at least one reading for limits")
    if variant == "p95":
        return percentile(xs, 5.0), percentile(xs, 95.0)
    if variant == "drop":
        s = sorted(xs)
        k = len(s) // 16
        keep = s[k: len(s) - k] or s
        return keep[0], keep[-1]
    if variant == "none":
        return min(xs), max(xs)
    raise ValueError("unknown variant %r" % variant)


def clip(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def hoeffding_halfwidth(rng_w: float, m: int, delta: float = DELTA) -> float:
    if m <= 0:
        return math.inf
    return rng_w * math.sqrt(math.log(2.0 / delta) / (2.0 * m))


def plan_window(rng_w: float, lsb: float, band: float = BAND_W, delta: float = DELTA):
    """Smallest even window n = 2m with h + lsb/2 <= band; None if lsb/2 >= band."""
    slack = band - lsb / 2.0
    if slack <= 0:
        return None
    if rng_w <= 0:
        return 2
    m = math.ceil(math.log(2.0 / delta) * rng_w * rng_w / (2.0 * slack * slack))
    return 2 * max(1, m)


# ---- the certified gate budget ----------------------------------------------------

def gate_budget(readings, lsb: float, variant: str):
    n = len(readings)
    if n < 4:
        raise ValueError("need at least 4 readings (2 for limits, 2 for the mean)")
    if lsb < 0:
        raise ValueError("lsb must be >= 0")
    half = n // 2
    first, second = list(readings[:half]), list(readings[half:])
    lo, hi = winsor_limits(first, variant)
    clipped = [clip(v, lo, hi) for v in second]
    m = len(clipped)
    mean_c = sum(clipped) / m
    h = hoeffding_halfwidth(hi - lo, m)
    q_err = lsb / 2.0
    lcb, ucb = mean_c - h - q_err, mean_c + h + q_err
    trit = 1 if lcb >= WATTS_MIN else (-1 if ucb < WATTS_MIN else 0)
    n_plan = plan_window(hi - lo, lsb)
    return {
        "variant": variant,
        "n": n,
        "limits": [round(lo, 6), round(hi, 6)],
        "clipped_frac": round(sum(1 for v in second if v < lo or v > hi) / m, 6),
        "mean_clip": round(mean_c, 6),
        "budget_w": {"quant": round(q_err, 6), "sampling": round(h, 6),
                     "total": round(h + q_err, 6), "band": BAND_W},
        "lcb": round(lcb, 6),
        "ucb": round(ucb, 6),
        "trit": trit,
        "band_sufficient": (h + q_err) <= BAND_W,
        "n_plan": n_plan,
    }


def evaluate(readings, lsb: float = 1.0, junior_ue5: bool = False, prev: str = GENESIS):
    rows = [gate_budget(readings, lsb, v) for v in VARIANTS]
    primary = rows[0]  # p95 is the gate's decision variant
    armed = bool(junior_ue5) and primary["trit"] == 1
    body = {
        "roadmap": ROADMAP_ITEM,
        "rate": RATE,
        "delta": DELTA,
        "watts_min": WATTS_MIN,
        "lsb": lsb,
        "junior_ue5": bool(junior_ue5),
        "measured": MEASURED,
        "rows": rows,
        "route": GATED_ROUTE if armed else DEFAULT_ROUTE,
        "os_route_launch_permitted": armed,
        "ue5_launch": UE5_LAUNCH,
        "bind": BIND,
        "ports": list(PORTS),
    }
    text = json.dumps(body, sort_keys=True, separators=(",", ":"))
    pk = pack([r["trit"] for r in rows])
    body["close"] = close_rule(prev, pk, text)
    body["prev"] = prev
    return body


# ---- selftest ---------------------------------------------------------------------

def _quantize(xs, lsb):
    if lsb <= 0:
        return list(xs)
    return [round(v / lsb) * lsb for v in xs]


def _series(rng, n, mu, sd, spike_p=0.0, spike=0.0):
    out = []
    for _ in range(n):
        v = rng.gauss(mu, sd)
        if spike_p and rng.random() < spike_p:
            v += spike if rng.random() < 0.5 else -spike
        out.append(v)
    return out


def selftest() -> int:
    rng = random.Random(20261017)
    checks = []

    def ok(name, cond):
        checks.append((name, bool(cond)))

    # pack / close-rule invariants
    tr = [1, 0, -1, 1, -1]
    ok("pack roundtrip", unpack(pack(tr), len(tr)) == tr)
    ok("pack I2_S codes", pack([0, 1, -1, 0]) == bytes([0b00011000]))
    a = close_rule(GENESIS, pack(tr), "x")
    ok("close deterministic", a == close_rule(GENESIS, pack(tr), "x") and len(a) == 16)
    ok("close chains", close_rule(a, pack(tr), "x") != a)

    # 1. steady 92 W with spikes -> certified above; armed only with JUNIOR_UE5=1
    hi_w = _quantize(_series(rng, 256, 92.0, 2.0, 0.04, 60.0), 1.0)
    r_on = evaluate(hi_w, 1.0, junior_ue5=True)
    r_off = evaluate(hi_w, 1.0, junior_ue5=False)
    ok("92W p95 trit +1", r_on["rows"][0]["trit"] == 1)
    ok("armed with flag", r_on["route"] == GATED_ROUTE and r_on["os_route_launch_permitted"])
    ok("not armed without flag", r_off["route"] == DEFAULT_ROUTE and not r_off["os_route_launch_permitted"])
    ok("ue5_launch false always", r_on["ue5_launch"] is False and UE5_LAUNCH is False)
    raw = r_on["rows"][2]
    ok("raw limits wider under spikes", raw["limits"][1] - raw["limits"][0] >
       r_on["rows"][0]["limits"][1] - r_on["rows"][0]["limits"][0])

    # 2. steady 68 W -> certified below
    lo_w = _quantize(_series(rng, 256, 68.0, 2.0), 1.0)
    ok("68W trit -1", evaluate(lo_w, 1.0, True)["rows"][0]["trit"] == -1)

    # 3. 80.3 W, short noisy window -> undecided, stays FrameForge2D
    mid = _quantize(_series(rng, 16, 80.3, 3.0), 1.0)
    r_mid = evaluate(mid, 1.0, True)
    ok("80.3W short window trit 0", r_mid["rows"][0]["trit"] == 0)
    ok("undecided stays default", r_mid["route"] == DEFAULT_ROUTE)

    # 4. quantization term is a hard bound: clipped mean of true vs quantized values
    worst = 0.0
    for lsb in (0.1, 0.5, 1.0, 2.0):
        for _ in range(50):
            true = _series(rng, 64, 85.0, 4.0, 0.05, 30.0)
            q = _quantize(true, lsb)
            lo, hi = winsor_limits(q[:32], "p95")
            mt = sum(clip(v, lo, hi) for v in true[32:]) / 32
            mq = sum(clip(v, lo, hi) for v in q[32:]) / 32
            worst = max(worst, abs(mt - mq) / (lsb / 2.0))
    ok("quant error <= lsb/2", worst <= 1.0 + 1e-9)

    # 5. Hoeffding coverage of the clipped-mean estimand at delta 0.05
    mu, sd, trials, miss = 83.0, 3.0, 300, 0
    for _ in range(trials):
        w = _series(rng, 64, mu, sd)
        lo, hi = winsor_limits(w[:32], "p95")
        # estimand E[clip(X, lo, hi)] by a large independent draw
        ref = _series(rng, 4000, mu, sd)
        est = sum(clip(v, lo, hi) for v in ref) / len(ref)
        mc = sum(clip(v, lo, hi) for v in w[32:]) / 32
        h = hoeffding_halfwidth(hi - lo, 32)
        if abs(mc - est) > h:
            miss += 1
    ok("coverage >= 1 - delta", miss / trials <= DELTA)

    # 6. sample plan: monotone in range and lsb, None when lsb/2 eats the band
    ok("plan monotone in range", plan_window(4.0, 1.0) <= plan_window(8.0, 1.0))
    ok("plan monotone in lsb", plan_window(6.0, 0.1) <= plan_window(6.0, 2.0))
    ok("plan none when lsb too coarse", plan_window(6.0, 8.0) is None)
    n_p = plan_window(6.0, 1.0)
    ok("plan meets band", hoeffding_halfwidth(6.0, n_p // 2) + 0.5 <= BAND_W + 1e-12)
    ok("plan is tight", hoeffding_halfwidth(6.0, n_p // 2 - 1) + 0.5 > BAND_W)

    # 7. close vector stable and chained through prev
    c1 = evaluate(hi_w, 1.0, True)["close"]
    ok("evaluate close stable", c1 == evaluate(hi_w, 1.0, True)["close"])
    ok("evaluate close chains", evaluate(hi_w, 1.0, True, prev=c1)["close"] != c1)

    # 8. standing flags
    ok("flags false", not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH or MEASURED))
    ok("bind loopback", BIND == "127.0.0.1")
    ok("rate 0.05", RATE == 0.05 and BAND_W == 4.0)
    ok("no socket import", "socket" not in globals())

    failed = [n for n, c in checks if not c]
    for n, c in checks:
        print(("PASS " if c else "FAIL ") + n)
    print(json.dumps({"selftest": "fail" if failed else "pass", "checks": len(checks),
                      "failed": failed, "close_92w": c1, "plan_r6_lsb1": n_p,
                      "coverage_miss": round(miss / trials, 4)}, sort_keys=True))
    return 1 if failed else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--watts", default="", help="comma list of watt readings")
    ap.add_argument("--lsb", type=float, default=1.0)
    ap.add_argument("--env-ue5", default="0")
    ap.add_argument("--prev", default=GENESIS)
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if not args.watts:
        ap.error("--watts is required unless --selftest")
    readings = [float(x) for x in args.watts.split(",") if x.strip()]
    print(json.dumps(evaluate(readings, args.lsb, args.env_ue5 == "1", args.prev),
                     sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
