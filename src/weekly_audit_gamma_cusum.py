#!/usr/bin/env python3
"""Weekly audit gamma CUSUM change-point check (chief weekly audit, web3node/stocksnode).

Roadmap anchor: JuniorHome docs/ROADMAP.md, Automations, "chief weekly audit"
(writes docs/WEEKLY_AUDIT.md). Quant Chief's pair A item for this slot names the
remaining engines web3node and stocksnode. This file adds the quant-error side
for those two engines from JuniorQuant: it does not write WEEKLY_AUDIT.md and it
does not touch any LLC path.

New capability vs earlier files: a one-sided Page CUSUM change-point detector
over the week's daily samples (reference = mean of days 0-2), per engine and per port. It answers "did the
gamma or the clip fraction shift at some day inside the week, and which day",
which a week-over-week drift number or a whole-week acceptance test cannot.
  * Gamma shift: two-sided CUSUM on log(gamma_day / gamma_ref) for absmean,
    junior_drop and p95, slack k = 0.025, alarm h = 0.15.
  * Clip-rate shift: Bernoulli CUSUM on daily P95 clip fraction against
    rate 0.05 (upward only, the side that breaks the i2sd ternary budget).
  * Ternary error: per-day max |w - q| over the analytic bound must be <= 1.

Ports touched (labels only): Home, OSai, FieldCore, Flagstaff, JuniorLLM.
Engines checked: web3node, stocksnode. i2sd userspace is named as
127.0.0.1:8767 only; no socket is opened, nothing binds.

Gammas (BitNet ternary, JuniorHome docs/WINSOR.md, QUANT_ERROR.md):
  absmean      gamma = mean |W|
  junior_drop  gamma = mean |W| after dropping the top 1/16 of |W|
  p95          gamma = 95th percentile of |W| (Winsor clip, rate 0.05)
Ternary: q = gamma * clamp(round(w / gamma), -1, 1).
Bound: |w - q| <= gamma/2 if |w| <= 1.5*gamma else |w| - gamma.

Close rule: sha256(prev || pack || text)[:16].
Flags: mint false, socket false, layer_one false, fetch false, ue5_launch false.
Stdlib only. Usage: --selftest | --input week.json [--prev HEX]
  week.json: {"engine": {"port": [[w, ...] per day, ...]}}
"""
import argparse
import hashlib
import json
import math
import random
import sys

RATE = 0.05
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
ENGINES = ("web3node", "stocksnode")
I2SD = "127.0.0.1:8767"
FLAGS = {"mint": False, "socket": False, "layer_one": False,
         "fetch": False, "ue5_launch": False}
K_GAMMA, H_GAMMA = 0.025, 0.15
REF_DAYS = 3  # reference gamma = mean of the first three days


def gamma_absmean(w):
    return sum(abs(x) for x in w) / len(w)


def gamma_junior_drop(w):
    a = sorted(abs(x) for x in w)
    keep = a[: len(a) - len(a) // 16] or a
    return sum(keep) / len(keep)


def gamma_p95(w):
    a = sorted(abs(x) for x in w)
    k = max(0, min(len(a) - 1, int(math.ceil(0.95 * len(a))) - 1))
    return a[k]


GAMMAS = {"absmean": gamma_absmean, "junior_drop": gamma_junior_drop,
          "p95": gamma_p95}


def ternary(x, g):
    if g <= 0:
        return 0.0
    return g * max(-1, min(1, round(x / g)))


def bound(x, g):
    ax = abs(x)
    return g / 2 if ax <= 1.5 * g else ax - g


def err_ratio(w, g):
    worst = 0.0
    for x in w:
        b = bound(x, g)
        e = abs(x - ternary(x, g))
        r = e / b if b > 0 else (0.0 if e <= 1e-12 else math.inf)
        worst = max(worst, r)
    return worst


def cusum2(xs, k, h):
    """Two-sided Page CUSUM. Returns (alarm_day or None, max_stat)."""
    sp = sn = top = 0.0
    for i, x in enumerate(xs):
        sp = max(0.0, sp + x - k)
        sn = max(0.0, sn - x - k)
        top = max(top, sp, sn)
        if sp > h or sn > h:
            return i, top
    return None, top


def bern_cusum(fracs, ns, p0=RATE, p1=2 * RATE, h=4.0):
    """Upward Bernoulli CUSUM (log-likelihood ratio p1 vs p0)."""
    a = math.log(p1 / p0)
    b = math.log((1 - p1) / (1 - p0))
    s = top = 0.0
    for i, (f, n) in enumerate(zip(fracs, ns)):
        c = f * n
        s = max(0.0, s + c * a + (n - c) * b)
        top = max(top, s)
        if s > h:
            return i, top
    return None, top


def port_check(days):
    out = {"days": len(days)}
    for name, fn in GAMMAS.items():
        gs = [fn(w) for w in days]
        base = gs[:REF_DAYS]
        ref = max(sum(base) / len(base), 1e-12)
        logs = [math.log(max(g, 1e-12) / ref) for g in gs]
        day, top = cusum2(logs, K_GAMMA, H_GAMMA)
        worst = max(err_ratio(w, g) for w, g in zip(days, gs))
        out[name] = {"alarm_day": day, "cusum_max": round(top, 6),
                     "err_over_bound": round(worst, 6)}
    fr, ns = [], []
    for w in days:
        g = gamma_p95(w)
        fr.append(sum(1 for x in w if abs(x) > g) / len(w))
        ns.append(len(w))
    day, top = bern_cusum(fr, ns)
    out["clip"] = {"rate": RATE, "alarm_day": day, "cusum_max": round(top, 6),
                   "max_frac": round(max(fr), 6)}
    ok = (out["clip"]["alarm_day"] is None and
          all(out[n]["alarm_day"] is None and out[n]["err_over_bound"] <= 1 + 1e-9
              for n in GAMMAS))
    out["verdict"] = "pass" if ok else "review"
    return out


def close(prev, pack, text):
    return hashlib.sha256(prev.encode() + pack.encode() + text.encode()
                          ).hexdigest()[:16]


def audit(week, prev="0" * 16):
    rep = {"item": "ROADMAP chief weekly audit", "i2sd": I2SD,
           "flags": FLAGS, "rate": RATE, "engines": {}}
    for eng in ENGINES:
        rep["engines"][eng] = {p: port_check(week[eng][p])
                               for p in PORTS if p in week.get(eng, {})}
    text = json.dumps(rep, sort_keys=True)
    rep["close"] = close(prev, "weekly_audit_gamma_cusum", text)
    return rep


def synth(rng, n=2048, days=7, shift_day=None, scale=1.6):
    out = []
    for d in range(days):
        s = scale if shift_day is not None and d >= shift_day else 1.0
        out.append([rng.gauss(0, 0.02 * s) for _ in range(n)])
    return out


def selftest():
    rng = random.Random(95)
    week = {e: {p: synth(rng) for p in PORTS} for e in ENGINES}
    rep = audit(week)
    for e in ENGINES:
        for p in PORTS:
            assert rep["engines"][e][p]["verdict"] == "pass", (e, p, rep["engines"][e][p])
    # a shift on day 4 must alarm on or after day 4
    week["web3node"]["OSai"] = synth(rng, shift_day=4)
    r2 = audit(week)
    c = r2["engines"]["web3node"]["OSai"]
    assert c["verdict"] == "review"
    assert c["absmean"]["alarm_day"] is not None and c["absmean"]["alarm_day"] >= 4
    # clip CUSUM: steady 5% stays quiet, 12% alarms
    assert bern_cusum([0.05] * 7, [512] * 7)[0] is None
    assert bern_cusum([0.12] * 7, [512] * 7)[0] is not None
    # bound holds for every gamma on heavy tails
    w = [rng.gauss(0, 1) * (8 if i % 50 == 0 else 1) for i in range(800)]
    for fn in GAMMAS.values():
        assert err_ratio(w, fn(w)) <= 1 + 1e-9
    # close rule is deterministic and chains
    rng2 = random.Random(7)
    wk = {e: {p: synth(rng2, n=256) for p in PORTS} for e in ENGINES}
    assert audit(wk)["close"] == audit(wk)["close"]
    assert audit(wk)["close"] != audit(wk, prev="f" * 16)["close"]
    a1 = close("0" * 16, "p", "t")
    assert a1 == close("0" * 16, "p", "t") and len(a1) == 16 and a1 != close(a1, "p", "t")
    assert all(v is False for v in FLAGS.values()) and I2SD.startswith("127.0.0.1:")
    print("selftest ok close=%s" % rep["close"])
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--input")
    ap.add_argument("--prev", default="0" * 16)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.input:
        ap.error("--input or --selftest required")
    with open(a.input) as f:
        rep = audit(json.load(f), a.prev)
    print(json.dumps(rep, indent=2, sort_keys=True))
    return 0 if all(v["verdict"] == "pass" for e in rep["engines"].values()
                    for v in e.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
