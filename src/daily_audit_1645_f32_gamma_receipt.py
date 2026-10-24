#!/usr/bin/env python3
"""JuniorQuant 16:45 daily audit: f32 gamma scalar receipt.

Roadmap anchor: JuniorHome docs/ROADMAP.md, "16:45 daily audit" item, the
JuniorQuant engine receipt that stores the day's f32 Winsor gamma scalars
(absmean, junior_drop, p95) as benched by JuniorLLM 544c158.

New capability: a canonical, bit-exact receipt of the three daily gammas.
Each scalar is stored as its IEEE-754 binary32 hex word (struct '<f'), so any
engine (BitNet-mlx f32, JuniorQuant f64, i2sd userspace) reproduces the same
bytes. The receipt also records the per-gamma clip fraction, the ternary
quant-error RMS, and the analytic bound, and chains the day to the previous
receipt with the close rule sha256(prev || pack || text)[:16].

Gammas (see JuniorHome docs/WINSOR.md, QUANT_ERROR.md):
  absmean      gamma = mean |W|                       (BitNet b1.58)
  junior_drop  gamma = mean |W| after dropping top 1/16 of |W|
  p95          gamma = 95th percentile of |W|, clip rate target 0.05

Ports touched (read-only, no network): Home, OSai, FieldCore, Flagstaff,
JuniorLLM. i2sd userspace is 127.0.0.1:8767; this file opens no socket.
Flags: mint false, socket false, layer_one false, fetch false,
ue5_launch false. Stdlib only. Run with --selftest.
"""
import hashlib
import json
import math
import random
import struct
import sys

RATE = 0.05
DROP_FRAC = 1.0 / 16.0
I2SD_BIND = "127.0.0.1:8767"
FLAGS = {"mint": False, "socket": False, "layer_one": False,
         "fetch": False, "ue5_launch": False}
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
GENESIS = "0" * 16


def f32(x):
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def f32_hex(x):
    return struct.pack("<f", float(x)).hex()


def gammas(weights):
    a = sorted(abs(float(w)) for w in weights)
    n = len(a)
    if n == 0:
        raise ValueError("empty weights")
    absmean = sum(a) / n
    keep = n - int(n * DROP_FRAC)
    keep = max(keep, 1)
    junior_drop = sum(a[:keep]) / keep
    # nearest-rank P95
    k = max(1, math.ceil(0.95 * n))
    p95 = a[k - 1]
    return {"absmean": f32(absmean), "junior_drop": f32(junior_drop),
            "p95": f32(p95)}


def ternary_stats(weights, gamma):
    """Clip fraction and RMS error of round(clip(W/g,-1,1))*g."""
    n = len(weights)
    g = gamma if gamma > 0 else 1e-12
    clipped = 0
    se = 0.0
    inside_se = 0.0
    for w in weights:
        r = w / g
        if abs(r) > 1.0:
            clipped += 1
        q = max(-1.0, min(1.0, round(r))) * g
        se += (w - q) ** 2
    rms = math.sqrt(se / n)
    return clipped / n, rms


def rms_bound(weights, gamma):
    """Analytic per-element bound: in-range error <= g/2, clipped error
    = |w| - g. Bound = sqrt(mean(min-case max)) taken elementwise."""
    g = gamma
    acc = 0.0
    for w in weights:
        a = abs(w)
        e = (a - g) if a > g else g / 2.0
        acc += e * e
    return math.sqrt(acc / len(weights))


def close(prev, pack, text):
    h = hashlib.sha256()
    h.update(prev.encode())
    h.update(pack.encode())
    h.update(text.encode())
    return h.hexdigest()[:16]


def build_receipt(day, weights, prev=GENESIS):
    gs = gammas(weights)
    body = {"engine": "JuniorQuant", "audit": "16:45 daily", "day": day,
            "bench_ref": "JuniorLLM 544c158", "rate": RATE,
            "n": len(weights), "i2sd": I2SD_BIND, "ports": list(PORTS),
            "flags": FLAGS, "gammas": {}}
    for name, g in gs.items():
        frac, rms = ternary_stats(weights, g)
        bound = rms_bound(weights, g)
        entry = {"f32_hex": f32_hex(g), "value": g,
                 "clip_frac": round(frac, 6), "rms": rms, "bound": bound,
                 "within_bound": rms <= bound + 1e-12}
        if name == "p95":
            entry["clip_ok"] = frac <= RATE + 1.0 / len(weights)
        body["gammas"][name] = entry
    pack = "".join(body["gammas"][k]["f32_hex"]
                   for k in ("absmean", "junior_drop", "p95"))
    text = json.dumps(body, sort_keys=True, separators=(",", ":"))
    body["prev"] = prev
    body["pack"] = pack
    body["close"] = close(prev, pack, text)
    return body, text


def verify_receipt(receipt, text):
    gs = receipt["gammas"]
    pack = "".join(gs[k]["f32_hex"] for k in ("absmean", "junior_drop", "p95"))
    if pack != receipt["pack"]:
        return False
    for k in gs:
        if struct.unpack("<f", bytes.fromhex(gs[k]["f32_hex"]))[0] != gs[k]["value"]:
            return False
    return close(receipt["prev"], pack, text) == receipt["close"]


def selftest():
    rng = random.Random(1645)
    w = [rng.gauss(0.0, 0.02) for _ in range(4096)]
    w += [rng.choice((-1, 1)) * rng.uniform(0.08, 0.2) for _ in range(64)]
    r1, t1 = build_receipt("2026-10-23", w)
    assert verify_receipt(r1, t1)
    g = r1["gammas"]
    assert g["junior_drop"]["value"] < g["absmean"]["value"] < g["p95"]["value"]
    for k in g:
        assert g[k]["within_bound"], k
        assert len(g[k]["f32_hex"]) == 8
    assert g["p95"]["clip_ok"]
    # determinism and chaining
    r1b, t1b = build_receipt("2026-10-23", list(w))
    assert r1b["close"] == r1["close"] and t1b == t1
    r2, t2 = build_receipt("2026-10-24", w[::-1], prev=r1["close"])
    assert verify_receipt(r2, t2) and r2["prev"] == r1["close"]
    assert r2["pack"] == r1["pack"]  # order-invariant gammas
    assert r2["close"] != r1["close"]
    # tamper detection
    bad = json.loads(json.dumps(r1))
    bad["gammas"]["p95"]["f32_hex"] = f32_hex(g["p95"]["value"] * 1.001)
    assert not verify_receipt(bad, t1)
    assert all(v is False for v in FLAGS.values())
    assert I2SD_BIND.startswith("127.0.0.1:")
    print("selftest ok close=%s pack=%s" % (r1["close"], r1["pack"]))
    return 0


def main(argv):
    if "--selftest" in argv:
        return selftest()
    data = json.load(sys.stdin)
    r, _ = build_receipt(data["day"], data["weights"],
                         data.get("prev", GENESIS))
    print(json.dumps(r, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
