"""Live beta steps 4-5 sequential quant-error gate (truncated Wald SPRT). Stdlib only.

Roadmap item: JuniorLLM docs/BETA_TO_OS.md "Live beta" step 4 (JuniorClimbs
`scripts/stonefield_stdlib_probe.py`, no FastAPI required) and step 5
(`python scripts/backfill_check.py` exits 0); docs/COMPILED_BACKLOG.md
"A - live beta suite". Quant Chief pair A, slot 2026-10-17 18:20 MT.
This is the quant-error / Winsor-bound side of that item. The step-4/5 trit
pack and I2_S key side belongs to Trit Pack & I2_S
(JuniorLLM scripts/live_beta_steps45_i2s_key.py).

Why: steps 4 and 5 are re-run by automations five times a day. One green run
says little, and "green on the last run" can flip on noise. The question the
live beta actually needs answered is "is the per-run failure rate at most
RATE (0.05)?", with a stated error probability, from the fewest runs.

New capability (not in winsor_night_drift.py, which gave fixed-sample
Hoeffding / Bernstein UCBs, nor in any earlier file): a sequential test.
  * Wald SPRT on per-run fail indicators, H0: p_fail = p0 = RATE (healthy)
    vs H1: p_fail = p1 = 5 * RATE (broken), alpha = beta = RATE.
    Boundaries A = (1 - beta) / alpha = 19, B = beta / (1 - alpha) = 1/19.
    The likelihood ratio (p1/p0)^k ((1-p1)/(1-p0))^(n-k) is evaluated in
    exact fractions.Fraction, never math.log, so the decision is identical on
    every libc / base (Debian, Alpine, macOS).
  * Truncation at N_MAX = 35 runs (5 runs/day x 7 days): undecided -> hold.
  * Exact operating characteristic of the TRUNCATED test by dynamic
    programming over (n, fails): true alpha = P(reject | p0), true beta =
    P(accept | p1), hold mass, and the average sample number (ASN) under both
    hypotheses. Wald's 19 / 1/19 bounds are approximations; this certifies the
    achieved error rates and asserts both are <= RATE.
  * Step-4 quant-error reading: a run of the stdlib probe reports the
    relative quantization error rel = SSE / sum(w^2) (JuniorHome
    docs/QUANT_ERROR.md) of ternary RoundClip(W / gamma, -1, 1) for the three
    Winsor gammas (JuniorHome docs/WINSOR.md):
        absmean = mean|W|                                (BitNet b1.58)
        drop    = mean|W| after dropping the top 1/16 of |W|   (Junior drop)
        p95     = mean of min(|W|, P95), nearest rank     (P95 clip)
    on a fixed seeded tensor. The run is a step-4 fail if its import audit is
    not clean or any gamma's rel drifts from this file's reference by more
    than RATE (relative). Reference uses math.fsum and random.random() only
    (Irwin-Hall, no gauss/log), so it is base-independent.
  * Step-5 reading: one exit code per run; nonzero is a fail.

Verdict trit per step: +1 accept H0 (fail rate <= RATE at error <= RATE),
-1 reject H0 (fail rate high), 0 continue / hold. Release fraction RATE
(0.05 cohort ramp step) only when both steps are +1, else 0.

It runs no probe, no backfill_check, no juniorctl, no test; no subprocess,
no network, no socket. It only reads text / JSON handed to it. The i2sd
userspace (127.0.0.1:8767) and bitnetd (127.0.0.1:8765) are referenced as the
consumers of these gammas; no port is opened. Bind 127.0.0.1 only.

Pack: I2_S 2-bit codes, MSB first: 0 -> 0b00, +1 -> 0b01, -1 -> 0b10.
Close rule: sha256(prev || pack || text)[:16]; prev = 16-hex ASCII (GENESIS
"0" * 16), pack = I2_S code of the verdict trits [step4, step5, release>0],
text = canonical verdict JSON (sorted keys, compact). Rate 0.05.
Compatible engines: BitNet-mlx, JuniorLLM, i2sd userspace, JuniorOS overlay.
Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM.
mint false, socket false, layer_one false, fetch false, ue5_launch false.
No model pull, no GGUF download, no hardware calls.

Usage:
  python3 src/live_beta_steps45_sprt_gate.py --selftest
  python3 src/live_beta_steps45_sprt_gate.py [--readings FILE.json]
      [--transcript FILE.txt] [--prev HEX16]
  readings JSON: [{"import_clean": true, "rel": {"absmean": x, "drop": y,
                   "p95": z}}, ...]   (one object per step-4 run, in order)
  transcript: lines like `exit=0`, `exit code: 1`, `returncode 0`
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from fractions import Fraction

RATE = 0.05
RATE_F = Fraction(1, 20)
P0 = RATE_F
P1 = 5 * RATE_F
ALPHA = RATE_F
BETA = RATE_F
UPPER = (1 - BETA) / ALPHA      # 19
LOWER = BETA / (1 - ALPHA)      # 1/19
N_MAX = 35
GENESIS = "0" * 16
SEED = 20261017
N_W = 4096
GAMMAS = ("absmean", "drop", "p95")
BIND = "127.0.0.1"
MINT = SOCKET = LAYER_ONE = FETCH = UE5_LAUNCH = False


# ---------------------------------------------------------------- pack / close
def pack_i2s(trits) -> bytes:
    code = {0: 0b00, 1: 0b01, -1: 0b10}
    out = bytearray()
    for i in range(0, len(trits), 4):
        b = 0
        chunk = list(trits[i:i + 4])
        for j, t in enumerate(chunk):
            b |= code[t] << (6 - 2 * j)
        out.append(b)
    return bytes(out)


def close_rule(prev: str, pack: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode() + pack + text.encode()).hexdigest()[:16]


# ---------------------------------------------------------------- SPRT
def likelihood_ratio(n: int, k: int) -> Fraction:
    return (P1 / P0) ** k * ((1 - P1) / (1 - P0)) ** (n - k)


def decide(n: int, k: int) -> int:
    lr = likelihood_ratio(n, k)
    if lr >= UPPER:
        return -1
    if lr <= LOWER:
        return 1
    return 0


def sprt(fails) -> dict:
    n = k = 0
    for f in fails:
        n += 1
        k += 1 if f else 0
        d = decide(n, k)
        if d != 0:
            return {"trit": d, "n": n, "fails": k, "stopped": True}
        if n >= N_MAX:
            break
    return {"trit": 0, "n": n, "fails": k, "stopped": False}


def operating_characteristic(p: Fraction) -> dict:
    """Exact P(accept), P(reject), P(hold) and ASN of the truncated SPRT."""
    p = float(p)
    cont = {0: 1.0}
    acc = rej = asn = 0.0
    for n in range(1, N_MAX + 1):
        nxt = {}
        for k, m in cont.items():
            asn += m
            for kk, pm in ((k + 1, m * p), (k, m * (1.0 - p))):
                d = decide(n, kk)
                if d == 1:
                    acc += pm
                elif d == -1:
                    rej += pm
                else:
                    nxt[kk] = nxt.get(kk, 0.0) + pm
        cont = nxt
    hold = math.fsum(cont.values())
    return {"accept": acc, "reject": rej, "hold": hold, "asn": asn}


def min_clean_runs() -> int:
    n = 1
    while decide(n, 0) != 1:
        n += 1
    return n


# ---------------------------------------------------------------- Winsor ref
def synthetic(n: int = N_W, seed: int = SEED):
    rng = random.Random(seed)
    w = []
    for i in range(n):
        z = sum(rng.random() for _ in range(12)) - 6.0   # Irwin-Hall ~ N(0,1)
        if i % 64 == 0:
            z *= 8.0                                    # sparse outliers
        w.append(z * 0.02)
    return w


def gamma_of(w, which: str) -> float:
    a = sorted(abs(x) for x in w)
    n = len(a)
    if which == "absmean":
        return math.fsum(a) / n
    if which == "drop":
        keep = n - n // 16
        return math.fsum(a[:keep]) / keep
    if which == "p95":
        c = a[max(0, math.ceil(0.95 * n) - 1)]
        return math.fsum(min(x, c) for x in a) / n
    raise ValueError(which)


def rel_error(w, which: str) -> float:
    g = gamma_of(w, which)
    sse = []
    for x in w:
        t = 0 if abs(x) < 0.5 * g else (1 if x > 0 else -1)
        sse.append((x - g * t) ** 2)
    return math.fsum(sse) / math.fsum(x * x for x in w)


def reference() -> dict:
    w = synthetic()
    return {g: rel_error(w, g) for g in GAMMAS}


def step4_fail(reading: dict, ref: dict) -> bool:
    if not reading.get("import_clean", False):
        return True
    rel = reading.get("rel", {})
    for g in GAMMAS:
        if g not in rel:
            return True
        if abs(float(rel[g]) - ref[g]) > RATE * ref[g]:
            return True
    return False


_EXIT = re.compile(r"(?:exit(?:\s*code)?|returncode)\s*[:=]?\s*(-?\d+)", re.I)


def parse_exits(text: str):
    return [int(m.group(1)) for m in _EXIT.finditer(text)]


# ---------------------------------------------------------------- evaluate
def evaluate(readings, exits, prev: str = GENESIS) -> dict:
    ref = reference()
    s4 = sprt([step4_fail(r, ref) for r in readings])
    s5 = sprt([e != 0 for e in exits])
    oc0 = operating_characteristic(P0)
    oc1 = operating_characteristic(P1)
    release = RATE if (s4["trit"] == 1 and s5["trit"] == 1) else 0.0
    body = {
        "item": "BETA_TO_OS live beta steps 4-5",
        "rate": RATE,
        "p0": str(P0), "p1": str(P1), "alpha": str(ALPHA), "beta": str(BETA),
        "n_max": N_MAX,
        "min_clean_runs": min_clean_runs(),
        "exact_alpha": round(oc0["reject"], 12),
        "exact_beta": round(oc1["accept"], 12),
        "hold_h0": round(oc0["hold"], 12),
        "hold_h1": round(oc1["hold"], 12),
        "asn_h0": round(oc0["asn"], 6),
        "asn_h1": round(oc1["asn"], 6),
        "reference_rel": {g: round(v, 12) for g, v in ref.items()},
        "step4": s4,
        "step5": s5,
        "release": release,
        "bind": BIND,
        "mint": MINT, "socket": SOCKET, "layer_one": LAYER_ONE,
        "fetch": FETCH, "ue5_launch": UE5_LAUNCH,
    }
    text = json.dumps(body, sort_keys=True, separators=(",", ":"))
    trits = [s4["trit"], s5["trit"], 1 if release > 0 else 0]
    body["trits"] = trits
    body["prev"] = prev
    body["close"] = close_rule(prev, pack_i2s(trits), text)
    return body


# ---------------------------------------------------------------- selftest
def selftest() -> int:
    bad = []

    def ok(name, cond):
        print(("PASS " if cond else "FAIL ") + name)
        if not cond:
            bad.append(name)

    ok("pack codes", pack_i2s([0, 1, -1, 0]) == bytes([0b00011000])
       and pack_i2s([1]) == b"\x40")
    exp = hashlib.sha256(GENESIS.encode() + b"\x40" + b"t").hexdigest()[:16]
    ok("close rule matches team form", close_rule(GENESIS, b"\x40", "t") == exp)
    ok("boundaries 19 and 1/19", UPPER == 19 and LOWER == Fraction(1, 19))
    # 13 clean runs: (15/19)^13 <= 1/19 < (15/19)^12
    ok("min clean runs = 13", min_clean_runs() == 13)
    ok("two early fails reject", sprt([True, True])["trit"] == -1)
    ok("one fail then clean needs more", sprt([True] + [False] * 13)["trit"] == 0
       and sprt([True] + [False] * 20)["trit"] == 1)
    ok("short clean run holds", sprt([False] * 5)["trit"] == 0)
    oc0, oc1 = operating_characteristic(P0), operating_characteristic(P1)
    ok("OC sums to 1 (H0)", abs(oc0["accept"] + oc0["reject"] + oc0["hold"] - 1) < 1e-12)
    ok("OC sums to 1 (H1)", abs(oc1["accept"] + oc1["reject"] + oc1["hold"] - 1) < 1e-12)
    ok("exact alpha <= RATE", oc0["reject"] <= RATE)
    ok("exact beta <= RATE", oc1["accept"] <= RATE)
    ok("ASN below N_MAX", oc0["asn"] < N_MAX and oc1["asn"] < N_MAX)
    # Monte Carlo cross-check of exact alpha (seeded, stdlib)
    rng = random.Random(7)
    trials = 20000
    rej = sum(1 for _ in range(trials)
              if sprt([rng.random() < 0.05 for _ in range(N_MAX)])["trit"] == -1)
    ok("MC alpha within 4 sigma of exact",
       abs(rej / trials - oc0["reject"]) <= 4 * math.sqrt(0.05 * 0.95 / trials) + 1e-3)
    ref = reference()
    ok("ref rel in (0,1)", all(0.0 < v < 1.0 for v in ref.values()))
    ok("Winsor gammas ordered drop,p95 <= absmean",
       gamma_of(synthetic(), "drop") <= gamma_of(synthetic(), "absmean")
       and gamma_of(synthetic(), "p95") <= gamma_of(synthetic(), "absmean"))
    ok("reference deterministic", reference() == ref)
    good = {"import_clean": True, "rel": dict(ref)}
    drift = {"import_clean": True, "rel": {g: v * 1.06 for g, v in ref.items()}}
    dirty = {"import_clean": False, "rel": dict(ref)}
    ok("step4 good passes", not step4_fail(good, ref))
    ok("step4 drift > RATE fails", step4_fail(drift, ref))
    ok("step4 FastAPI import fails", step4_fail(dirty, ref))
    ok("step4 missing gamma fails", step4_fail({"import_clean": True, "rel": {}}, ref))
    ok("parse exits", parse_exits("exit=0\nexit code: 1\nreturncode 0\n") == [0, 1, 0])
    r = evaluate([good] * 13, [0] * 13)
    ok("both accepted -> release RATE", r["trits"] == [1, 1, 1] and r["release"] == RATE)
    r2 = evaluate([good] * 13, [1, 1])
    ok("step5 rejected -> held", r2["trits"] == [1, -1, 0] and r2["release"] == 0.0)
    ok("close 16 hex", len(r["close"]) == 16 and int(r["close"], 16) >= 0)
    ok("close deterministic", evaluate([good] * 13, [0] * 13)["close"] == r["close"])
    ok("close prev-sensitive", evaluate([good] * 13, [0] * 13, "f" * 16)["close"] != r["close"])
    ok("flags false, bind loopback", not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH)
       and BIND == "127.0.0.1")
    print("selftest", "FAIL" if bad else "PASS", f"{len(bad)} failed")
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--readings")
    ap.add_argument("--transcript")
    ap.add_argument("--prev", default=GENESIS)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not re.fullmatch(r"[0-9a-f]{16}", a.prev):
        print("prev must be 16 lowercase hex", file=sys.stderr)
        return 2
    readings = []
    if a.readings:
        with open(a.readings, encoding="utf-8") as fh:
            readings = json.load(fh)
    exits = []
    if a.transcript:
        with open(a.transcript, encoding="utf-8") as fh:
            exits = parse_exits(fh.read())
    print(json.dumps(evaluate(readings, exits, a.prev), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
