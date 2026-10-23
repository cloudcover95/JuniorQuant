#!/usr/bin/env python3
"""Weekly audit quant-error rollup (chief weekly audit, quant-error side).

Roadmap anchor: JuniorHome docs/ROADMAP.md, Automations, "chief weekly audit"
(Sun/Mon 17:00 MT, writes docs/WEEKLY_AUDIT.md). This file adds the quant-error
and Winsor-gamma section that the weekly audit had no file for. It also reports
the slot mismatch between docs/AUTOMATION_AUDIT.md (Sat 10:00/11:00 + Mon 17:00)
and docs/ROADMAP.md (Sun/Mon 17:00) as a finding; it does not pick a winner.

New capability vs earlier daily files: a seven-day rollup per port with
week-over-week gamma drift, worst-day ternary error vs the analytic bound, and
the Winsor rate check (rate 0.05) aggregated over the week.

Ports touched (read-only, as labels): Home, OSai, FieldCore, Flagstaff,
JuniorLLM. i2sd userspace is referenced at 127.0.0.1:8767 only; no socket is
opened.

Gammas (BitNet ternary, see JuniorHome docs/WINSOR.md, QUANT_ERROR.md):
  absmean      gamma = mean |W|
  junior_drop  gamma = mean |W| after dropping the top 1/16 of |W|
  p95          gamma = 95th percentile of |W| (Winsor clip, rate 0.05)
Ternary: q = gamma * clamp(round(w / gamma), -1, 1).
Bound per element: |w - q| <= gamma/2 if |w| <= 1.5*gamma else |w| - gamma.

Close rule: sha256(prev || pack || text)[:16].
Flags: mint false, socket false, layer_one false, fetch false, ue5_launch false.
Stdlib only. Usage: --selftest | --input week.json [--prev HEX]
"""
import argparse
import hashlib
import json
import math
import random
import sys

RATE = 0.05
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
I2SD = "127.0.0.1:8767"
FLAGS = {"mint": False, "socket": False, "layer_one": False,
         "fetch": False, "ue5_launch": False}
SLOTS = {"AUTOMATION_AUDIT.md": ["Sat 10:00", "Sat 11:00", "Mon 17:00"],
         "ROADMAP.md": ["Sun 17:00", "Mon 17:00"]}
DRIFT_TOL = 0.10  # week-over-week relative gamma drift that gets flagged


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


def day_stats(w):
    out = {}
    for name, fn in GAMMAS.items():
        g = fn(w)
        errs = [abs(x - ternary(x, g)) for x in w]
        viol = sum(1 for x, e in zip(w, errs) if e > bound(x, g) + 1e-12)
        out[name] = {"gamma": g,
                     "mse": sum(e * e for e in errs) / len(w),
                     "max_err": max(errs),
                     "bound_violations": viol}
    gp = out["p95"]["gamma"]
    out["winsor_rate"] = sum(1 for x in w if abs(x) > gp) / len(w)
    return out


def rollup(week):
    """week: {port: [day0_weights, ..., day6_weights]} -> report dict."""
    rep = {"ports": {}, "findings": []}
    for port in PORTS:
        days = week.get(port)
        if not days:
            rep["findings"].append("%s: no weekly data" % port)
            continue
        stats = [day_stats(d) for d in days]
        p = {}
        for name in GAMMAS:
            gs = [s[name]["gamma"] for s in stats]
            drift = (gs[-1] - gs[0]) / gs[0] if gs[0] else 0.0
            p[name] = {"gamma_first": gs[0], "gamma_last": gs[-1],
                       "drift": drift,
                       "worst_mse": max(s[name]["mse"] for s in stats),
                       "violations": sum(s[name]["bound_violations"]
                                         for s in stats)}
            if abs(drift) > DRIFT_TOL:
                rep["findings"].append("%s %s gamma drift %+.3f" %
                                       (port, name, drift))
            if p[name]["violations"]:
                rep["findings"].append("%s %s bound violations %d" %
                                       (port, name, p[name]["violations"]))
        rates = [s["winsor_rate"] for s in stats]
        p["winsor_rate_week"] = sum(rates) / len(rates)
        if p["winsor_rate_week"] > RATE + 1e-9:
            rep["findings"].append("%s winsor rate %.4f > %.2f" %
                                   (port, p["winsor_rate_week"], RATE))
        rep["ports"][port] = p
    a, r = set(SLOTS["AUTOMATION_AUDIT.md"]), set(SLOTS["ROADMAP.md"])
    if a != r:
        rep["findings"].append(
            "slot mismatch: AUTOMATION_AUDIT.md %s vs ROADMAP.md %s"
            % (sorted(a), sorted(r)))
    rep["i2sd"] = I2SD
    rep["flags"] = FLAGS
    rep["rate"] = RATE
    return rep


def close(prev, pack, text):
    return hashlib.sha256(prev.encode() + pack.encode() + text.encode()
                          ).hexdigest()[:16]


def render(rep):
    lines = ["## Quant error (weekly rollup)", ""]
    lines.append("| port | gamma | first | last | drift | worst mse | viol |")
    lines.append("|---|---|---|---|---|---|---|")
    for port, p in rep["ports"].items():
        for name in GAMMAS:
            s = p[name]
            lines.append("| %s | %s | %.5f | %.5f | %+.3f | %.6f | %d |" % (
                port, name, s["gamma_first"], s["gamma_last"], s["drift"],
                s["worst_mse"], s["violations"]))
    lines.append("")
    for f in rep["findings"] or ["none"]:
        lines.append("- " + f)
    return "\n".join(lines) + "\n"


def selftest():
    rng = random.Random(5)
    week = {}
    for i, port in enumerate(PORTS):
        scale = 0.02 * (1 + i)
        week[port] = [[rng.gauss(0, scale * (1 + 0.01 * d)) for _ in range(512)]
                      for d in range(7)]
    rep = rollup(week)
    assert set(rep["ports"]) == set(PORTS)
    for p in rep["ports"].values():
        for name in GAMMAS:
            assert p[name]["violations"] == 0, name
        assert abs(p["winsor_rate_week"] - RATE) < 0.01
        assert p["junior_drop"]["gamma_last"] < p["absmean"]["gamma_last"]
        assert p["p95"]["gamma_last"] > p["absmean"]["gamma_last"]
    assert any("slot mismatch" in f for f in rep["findings"])
    # drift detection
    week2 = dict(week)
    week2["OSai"] = week["OSai"][:6] + [[x * 1.5 for x in week["OSai"][6]]]
    assert any(f.startswith("OSai") and "drift" in f
               for f in rollup(week2)["findings"])
    # missing port
    w3 = {k: v for k, v in week.items() if k != "Flagstaff"}
    assert "Flagstaff: no weekly data" in rollup(w3)["findings"]
    assert all(v is False for v in FLAGS.values())
    assert I2SD.startswith("127.0.0.1:")
    text = render(rep)
    pack = json.dumps(rep, sort_keys=True, default=str)
    c = close("0" * 16, pack, text)
    assert len(c) == 16 and c == close("0" * 16, pack, text)
    assert c != close("1" * 16, pack, text)
    print("selftest ok close=%s" % c)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--input")
    ap.add_argument("--prev", default="0" * 16)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.input:
        ap.error("--input or --selftest required")
    with open(a.input) as fh:
        week = json.load(fh)
    rep = rollup(week)
    text = render(rep)
    sys.stdout.write(text)
    print("close %s" % close(a.prev, json.dumps(rep, sort_keys=True,
                                                 default=str), text))
    return 0


if __name__ == "__main__":
    sys.exit(main())
