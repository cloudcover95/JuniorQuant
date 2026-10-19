"""Live beta step 4: exact ternary-optimality gap of the StoneField holds_matrix seed.

Stdlib only, offline, read-only. Ports touched: Home, OSai, FieldCore,
Flagstaff, JuniorLLM (the seed is suite JuniorOSai, the walk reports port
JuniorBitNetFieldCore, node flagstaff-mtn is the Flagstaff field, the engine
conventions come from JuniorLLM junior_bitnet, the report lands for Home).
TCP ports named, never dialed: bitnetd 127.0.0.1:8765, i2sd userspace
127.0.0.1:8767. Bind 127.0.0.1 only; this file opens no socket.
mint false, socket false, layer_one false, fetch false, ue5_launch false.
No FastAPI, no model pull, no GGUF, no network, no subprocess.

Roadmap item: JuniorLLM docs/BETA_TO_OS.md "Live beta" step 4, JuniorClimbs
`scripts/stonefield_stdlib_probe.py` (no FastAPI required); docs/COMPILED_BACKLOG.md
"A - live beta suite". Quant Chief pair A, slot 2026-10-19 10:20 MT
("Live beta: remaining gaps only, prefer steps 4 and 5 if still unfiled").
This is the quant-error / Winsor-bound side of step 4 only. Earlier step 4-5
files gate run counts (JuniorQuant live_beta_steps45_sprt_gate.py) and
stdlib-import / backfill exit trits (BitNet-mlx live_beta_steps45_i2s_key.py);
neither looks at the numbers the probe actually ternarizes. The I2_S key side
belongs to Trit Pack & I2_S. Step 3 receipts, security, night/ask/health,
step 6 and LLC paths are out of scope.

The gap: the stdlib StoneField probe walks a public seed node and ternarizes
its holds_matrix (row-major flatten) with a Winsor gamma, then packs I2_S.
Nothing says how far that trit pack is from the best ternary code those same
numbers admit, whether the three Winsor gammas even differ on a 6-9 entry
matrix, or whether a float engine and an exact reader agree on every trit.

New capability (not in any earlier pair-A file): the exact global ternary
optimum and a three-term split of each Winsor gamma's error against it.
  * exact reading: the seed JSON is parsed with decimals as fractions.Fraction,
    so 0.8 means 4/5, not the nearest double.
  * gammas, matched to the engines (JuniorHome docs/WINSOR.md):
      absmean  mean|W|                               eps 0, zero -> 1.0
      drop     mean |W| of the smallest n - floor(n/16)  eps 0, zero -> 1.0
      clip     mean min(|W|, tau), tau = sorted|W|[floor(0.95 (n-1))]
               (the JuniorLLM junior_bitnet/winsor.py index), eps 1e-7
    t_i = clip(round(w_i / (gamma + eps)), -1, 1), round half-even (Python).
    clip is the gamma the walk itself uses (JuniorLLM ports/gaia_walk.py calls
    junior_bitnet.winsor.pack); absmean and drop are the other two engines.
  * optimum: min over alpha >= 0 and t in {-1,0,1}^n of ||W - alpha t||^2.
    For a support of size k the best support is the k largest |w| and the best
    alpha is their mean, so SSE_opt = E - max_k P_k^2 / k with E = ||W||^2 and
    P_k the sum of the k largest |w|. Exact, O(n log n), attained.
  * split, exact identity for the support K (k = |K|, a = sum_K |w| / k):
      SSE_gamma = SSE_opt + (SSE_K - SSE_opt) + k (gamma - a)^2
                  irreducible   support excess    scale excess
    with SSE_K = E - (sum_K |w|)^2 / k. excess = SSE_gamma / SSE_opt - 1.
  * variant trit: +1 excess <= RATE; 0 support excess <= RATE * SSE_opt but
    the scale is off (storing a instead of gamma fixes it with the identical
    I2_S pack, no trit changes); -1 the trits themselves are off by > RATE.
  * collapse: drop equals absmean whenever floor(n/16) = 0 (n < 16); the file
    reports, per node, which gammas coincide exactly, so a three-variant claim
    on a tiny seed matrix is not mistaken for three measurements.
  * tie trit: an entry with |w| / (gamma + eps) exactly 1/2 rounds to 0 under
    half-even (Python, numpy) but to +-1 under C roundf (half away from zero);
    any exact tie, or any float-replica trit that disagrees with the exact
    trit, makes the tie trit -1 (engines can disagree); else +1.
  * shape trit: +1 rectangular finite matrix; 0 no holds_matrix (the walk falls
    back to a hold-name heuristic that is not a weight tensor, no claim made);
    -1 ragged, empty, or non-numeric.
Rel error is reported as rel_l2 = sqrt(SSE / E) (JuniorHome docs/QUANT_ERROR.md
quotes trit rel_l2 ~0.54 for a Gaussian n=128; tiny seed matrices differ).
Only public nodes are read (covenant scrape false, private_land false).
On the current public seed this finds: every node's absmean/drop trits already
sit on the optimal support (only the scale is off, trit 0), drop equals absmean
on all three nodes (n = 6..9 < 16), and on xanadu the clip gamma flips -0.5 to
-1, a support 75% worse than optimal (trit -1, exit 1). That is a finding about
the seed and the clip gamma, reported, never patched.

It does not import, wrap, test, or re-derive any team file or earlier Winsor
file (JuniorQuant src/winsor_clip.py, src/i2sd_*, src/live_beta_*; JuniorStock
risk/winsor_quant_error.py, security/winsor_outlier_health.py,
quant/calibration/*.py). It only reads a seed JSON you name.

Pack: I2_S 2-bit codes 0->0, +1->1, -1->2, four per byte MSB first, zero padded
(JuniorLLM junior_bitnet/i2s.py). Close rule: key = sha256(prev || pack ||
text)[:16], prev as ASCII hex, GENESIS "0" * 16, chained once per (node,
variant) with pack = that variant's trits and text = canonical row JSON, then
closed over the summary with pack = every absmean pack concatenated. Rate lock:
sha256(prev || pack || b"0.05").hexdigest() beside the final key. RATE = 0.05.

Exit: 0 no -1 trit; 1 some -1 trit; 2 unreadable input.
Run: python3 src/live_beta_step4_stonefield_ternary_gap.py --selftest
     python3 src/live_beta_step4_stonefield_ternary_gap.py --seed junior_osai/goldens/stonefield.json
     python3 src/live_beta_step4_stonefield_ternary_gap.py --matrix "0.8,-1.2,2.5;0.1,0.9,-0.4"
Compatible engines: JuniorLLM junior_bitnet (winsor.pack, math.absmean,
i2s.pack), BitNet-mlx, bitnetd, i2sd userspace, JuniorQuant, JuniorStock.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from fractions import Fraction as F
from typing import Optional

RATE = F(1, 20)
RATE_TEXT = "0.05"
GENESIS = "0" * 16
VARIANTS = ("absmean", "drop", "clip")
EPS = {"absmean": F(0), "drop": F(0), "clip": F(1, 10 ** 7)}
EPS_FLOAT = {"absmean": 0.0, "drop": 0.0, "clip": 1e-7}
CLIP_PCT = 95.0
BIND = "127.0.0.1"
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False,
         "ue5_launch": False, "fastapi": False, "model_pull": False}


# ------------------------------------------------------------------ helpers
def pack(trits: list[int]) -> bytes:
    acc = bits = 0
    out = bytearray()
    for t in trits:
        if t not in (-1, 0, 1):
            raise ValueError("trit must be -1, 0 or 1")
        acc = (acc << 2) | (0 if t == 0 else (1 if t == 1 else 2))
        bits += 2
        if bits == 8:
            out.append(acc)
            acc = bits = 0
    if bits:
        out.append(acc << (8 - bits))
    return bytes(out)


def close_rule(prev: str, pk: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode() + pk + text.encode()).hexdigest()[:16]


def rate_lock(prev: str, pk: bytes) -> str:
    return hashlib.sha256(prev.encode() + pk + RATE_TEXT.encode()).hexdigest()


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def fx(q: Optional[F], nd: int = 12):
    """Fraction -> rounded float for reports (exact values stay internal)."""
    if q is None:
        return None
    return round(float(q), nd)


def clip_index_float(n: int) -> int:
    # the JuniorLLM junior_bitnet/winsor.py _pct expression, kept in float
    return min(n - 1, max(0, math.floor((CLIP_PCT / 100.0) * (n - 1))))


def clip_index_exact(n: int) -> int:
    return min(n - 1, max(0, (95 * (n - 1)) // 100))


def round_half_even(q: F) -> int:
    return round(q)  # Fraction.__round__ is half-even, like Python float round


# ------------------------------------------------------------------ gammas
def gamma_of(variant: str, xs: list[F]) -> F:
    a = sorted(abs(x) for x in xs)
    n = len(a)
    if variant == "absmean":
        g = sum(a, F(0)) / n
    elif variant == "drop":
        m = n - n // 16
        g = sum(a[:m], F(0)) / m
    elif variant == "clip":
        tau = a[clip_index_exact(n)]
        g = sum((min(v, tau) for v in a), F(0)) / n
    else:
        raise ValueError(variant)
    if variant != "clip" and g == 0:
        g = F(1)  # junior_bitnet.math.absmean: mean_abs(xs) or 1.0
    return g


def quantize(xs: list[F], g: F, eps: F) -> tuple[list[int], int]:
    d = g + eps
    if d == 0:
        return [0] * len(xs), 0
    trits, ties = [], 0
    for x in xs:
        r = x / d
        if abs(r) == F(1, 2):
            ties += 1
        trits.append(max(-1, min(1, round_half_even(r))))
    return trits, ties


def float_replica(variant: str, xs: list[F]) -> list[int]:
    w = [float(x) for x in xs]
    a = sorted(abs(v) for v in w)
    n = len(a)
    if variant == "absmean":
        g = (sum(abs(v) for v in w) / n) or 1.0
    elif variant == "drop":
        m = n - n // 16
        g = (sum(a[:m]) / m) or 1.0
    else:
        tau = a[clip_index_float(n)]
        g = sum(min(v, tau) for v in (abs(u) for u in w)) / n
    d = g + EPS_FLOAT[variant]
    if d == 0:
        return [0] * n
    return [max(-1, min(1, int(round(v / d)))) for v in w]


# ------------------------------------------------------------------ optimum
def ternary_optimum(xs: list[F]) -> dict:
    a = sorted((abs(x) for x in xs), reverse=True)
    e = sum((x * x for x in xs), F(0))
    best_k, best_v, p = 0, F(0), F(0)
    for k, v in enumerate(a, start=1):
        p += v
        val = p * p / k
        if val > best_v:
            best_k, best_v = k, val
    return {"E": e, "k": best_k, "sse": e - best_v,
            "alpha": (sum(a[:best_k], F(0)) / best_k) if best_k else F(0)}


def split(xs: list[F], trits: list[int], g: F, opt: dict) -> dict:
    e = opt["E"]
    sse_g = sum(((x - g * t) ** 2 for x, t in zip(xs, trits)), F(0))
    k = sum(1 for t in trits if t)
    sk_abs = sum((abs(x) for x, t in zip(xs, trits) if t), F(0))
    sign_ok = all((t == 0) or (t > 0) == (x > 0) for x, t in zip(xs, trits))
    if k:
        a = sk_abs / k
        sse_k = e - sk_abs * sk_abs / k
        scale_ex = k * (g - a) ** 2
    else:
        a, sse_k, scale_ex = F(0), e, F(0)
    support_ex = sse_k - opt["sse"]
    if sign_ok:
        assert sse_g == opt["sse"] + support_ex + scale_ex, "split identity"
    assert support_ex >= 0 and sse_g >= opt["sse"], "optimum is a floor"
    so = opt["sse"]
    if so == 0:
        excess = F(0) if sse_g == 0 else None
        sup_r = F(0) if support_ex == 0 else None
        sca_r = F(0) if scale_ex == 0 else None
    else:
        excess, sup_r, sca_r = sse_g / so - 1, support_ex / so, scale_ex / so
    if excess is not None and excess <= RATE:
        trit = 1
    elif sup_r is not None and sup_r <= RATE:
        trit = 0
    else:
        trit = -1
    return {"sse": sse_g, "k": k, "refit_alpha": a, "support_excess": support_ex,
            "scale_excess": scale_ex, "excess": excess, "support_ratio": sup_r,
            "scale_ratio": sca_r, "trit": trit, "sign_ok": sign_ok}


# ------------------------------------------------------------------ nodes
def shape_of(m) -> tuple[int, Optional[list[F]], str]:
    if m is None:
        return 0, None, "no holds_matrix (hold-name fallback, not a weight tensor)"
    if not isinstance(m, list) or not m:
        return -1, None, "holds_matrix empty or not a list"
    width = None
    xs: list[F] = []
    for row in m:
        if not isinstance(row, list) or not row:
            return -1, None, "empty or non-list row"
        if width is None:
            width = len(row)
        elif len(row) != width:
            return -1, None, "ragged rows"
        for v in row:
            if isinstance(v, bool) or not isinstance(v, (F, int)):
                return -1, None, "non-numeric entry"
            xs.append(F(v))
    return 1, xs, "%dx%d" % (len(m), width)


def certify_matrix(xs: list[F]) -> dict:
    n = len(xs)
    opt = ternary_optimum(xs)
    out = {"n": n, "E": fx(opt["E"]), "opt_k": opt["k"], "opt_alpha": fx(opt["alpha"]),
           "opt_sse": fx(opt["sse"]),
           "opt_rel_l2": None if opt["E"] == 0 else round(math.sqrt(opt["sse"] / opt["E"]), 12),
           "variants": {}, "packs": {}}
    gammas, ties_any, mismatch_any = {}, 0, 0
    for v in VARIANTS:
        g = gamma_of(v, xs)
        gammas[v] = g
        trits, ties = quantize(xs, g, EPS[v])
        rep = float_replica(v, xs)
        mism = sum(1 for a, b in zip(trits, rep) if a != b)
        ties_any += ties
        mismatch_any += mism
        s = split(xs, trits, g, opt)
        out["packs"][v] = trits
        out["variants"][v] = {
            "gamma": fx(g), "eps": fx(EPS[v], 9), "trits": trits, "k": s["k"],
            "sse": fx(s["sse"]),
            "rel_l2": None if opt["E"] == 0 else round(math.sqrt(s["sse"] / opt["E"]), 12),
            "excess": fx(s["excess"]), "support_ratio": fx(s["support_ratio"]),
            "scale_ratio": fx(s["scale_ratio"]), "refit_alpha": fx(s["refit_alpha"]),
            "ties": ties, "float_mismatch": mism, "trit": s["trit"],
            "i2s_hex": pack(trits).hex()}
    out["collapse"] = {"drop_eq_absmean": gammas["drop"] == gammas["absmean"],
                       "clip_eq_absmean": gammas["clip"] == gammas["absmean"],
                       "drop_eq_clip": gammas["drop"] == gammas["clip"],
                       "drop_distinct_needs_n_ge": 16,
                       "clip_index_exact": clip_index_exact(n),
                       "clip_index_float": clip_index_float(n)}
    if clip_index_exact(n) != clip_index_float(n):
        mismatch_any += 1
    out["tie_trit"] = -1 if (ties_any or mismatch_any) else 1
    return out


def certify_seed(seed: dict, prev: str = GENESIS) -> dict:
    cov = seed.get("covenant") or {}
    rows, skipped, all_abs = [], [], b""
    for node in seed.get("nodes") or []:
        nid = str(node.get("id", ""))
        if not node.get("public"):
            skipped.append(nid)
            continue
        st, xs, why = shape_of(node.get("holds_matrix"))
        row = {"id": nid, "shape_trit": st, "shape": why}
        packs = {}
        if xs is not None:
            cert = certify_matrix(xs)
            packs = cert.pop("packs")
            row.update(cert)
        if packs:
            for v in VARIANTS:
                vr = row["variants"][v]
                prev = close_rule(prev, pack(packs[v]), canon({"id": nid, "variant": v, **vr}))
                vr["key"] = prev
            all_abs += pack(packs["absmean"])
        else:
            prev = close_rule(prev, b"", canon(row))
            row["key"] = prev
        trits = [st] + ([row["tie_trit"]] + [row["variants"][v]["trit"] for v in VARIANTS]
                        if packs else [])
        row["trits"] = trits
        rows.append(row)
    worst = min((t for r in rows for t in r["trits"]), default=0)
    summary = {"item": "BETA_TO_OS Live beta step 4 stonefield stdlib probe, quant-error side",
               "rate": RATE_TEXT, "bind": BIND, "ports": list(PORTS), "flags": FLAGS,
               "covenant": {"scrape": bool(cov.get("scrape", False)),
                            "private_land": bool(cov.get("private_land", False))},
               "nodes": len(rows), "skipped_private": skipped, "worst_trit": worst}
    if summary["covenant"]["scrape"] or summary["covenant"]["private_land"]:
        worst = summary["worst_trit"] = -1
    key = close_rule(prev, all_abs, canon(summary))
    return {"summary": summary, "rows": rows, "key": key, "rate_lock": rate_lock(prev, all_abs),
            "rc": 1 if worst < 0 else 0}


def parse_seed(text: str) -> dict:
    return json.loads(text, parse_float=F, parse_int=F)


def parse_matrix(spec: str) -> list[list[F]]:
    return [[F(c.strip()) for c in r.split(",") if c.strip()] for r in spec.split(";") if r.strip()]


# ------------------------------------------------------------------ selftest
FIXTURE = """{
  "covenant": {"scrape": false, "private_land": false},
  "nodes": [
    {"id": "flagstaff-mtn", "public": true, "holds_matrix": [[0.8, -1.2, 2.5], [0.1, 0.9, -0.4], [1.5, -0.7, 0.2]]},
    {"id": "xanadu", "public": true, "holds_matrix": [[2.1, 0.4, -1.8], [-0.5, 1.1, 0.3]]},
    {"id": "golden-clear-creek", "public": true, "holds_matrix": [[0.6, 0.8, -0.2], [-1.0, 1.4, 0.5]]},
    {"id": "names-only", "public": true, "holds": ["jug"]},
    {"id": "ragged", "public": true, "holds_matrix": [[1.0, 2.0], [3.0]]},
    {"id": "private-crag", "public": false, "holds_matrix": [[9.0]]}
  ]
}"""


def brute_optimum(xs: list[F]) -> F:
    # every support of the k largest is not assumed here: try all 3^n codes
    import itertools
    best = None
    for code in itertools.product((-1, 0, 1), repeat=len(xs)):
        k = sum(1 for c in code if c)
        if k == 0:
            val = sum((x * x for x in xs), F(0))
        else:
            dot = sum((x * c for x, c in zip(xs, code)), F(0))
            if dot <= 0:
                continue
            al = dot / k
            val = sum(((x - al * c) ** 2 for x, c in zip(xs, code)), F(0))
        best = val if best is None or val < best else best
    return best


def selftest() -> str:
    ok = 0

    def check(name, cond):
        nonlocal ok
        if not cond:
            raise AssertionError(name)
        ok += 1

    check("close rule", close_rule(GENESIS, b"\x02", "x")
          == hashlib.sha256(b"0" * 16 + b"\x02x").hexdigest()[:16])
    check("i2s pack", pack([0, 1, -1, 0, 1]) == bytes([0b00011000, 0b01000000]))
    check("exact decimal", parse_seed('{"a": 0.1}')["a"] == F(1, 10))
    # optimum matches brute force over all 3^n codes
    import random
    rng = random.Random(5)
    for _ in range(40):
        n = rng.randint(1, 6)
        xs = [F(rng.randint(-40, 40), 10) for _ in range(n)]
        check("opt == brute", ternary_optimum(xs)["sse"] == brute_optimum(xs))
        opt = ternary_optimum(xs)
        for v in VARIANTS:
            g = gamma_of(v, xs)
            t, _ = quantize(xs, g, EPS[v])
            s = split(xs, t, g, opt)
            check("floor holds", s["sse"] >= opt["sse"])
    # collapse: n < 16 means drop == absmean exactly
    xs = [F(i, 3) for i in range(1, 10)]
    check("drop collapses", gamma_of("drop", xs) == gamma_of("absmean", xs))
    xs16 = [F(i) for i in range(1, 17)]
    check("drop distinct n=16", gamma_of("drop", xs16) < gamma_of("absmean", xs16))
    # exact tie: |w| = gamma / 2 is flagged (half-even vs roundf disagree)
    tie = certify_matrix([F(1), F(3), F(-2), F(-2)])  # absmean 2, |1|/2 = 1/2
    check("tie flagged", tie["variants"]["absmean"]["ties"] >= 1 and tie["tie_trit"] == -1)
    # fixture: the public seed values
    rep = certify_seed(parse_seed(FIXTURE))
    rows = {r["id"]: r for r in rep["rows"]}
    check("private skipped", rep["summary"]["skipped_private"] == ["private-crag"])
    check("names-only shape 0", rows["names-only"]["trits"] == [0])
    check("ragged shape -1", rows["ragged"]["trits"] == [-1] and rep["rc"] == 1)
    fl = rows["flagstaff-mtn"]
    check("flagstaff 3x3", fl["shape"] == "3x3" and fl["n"] == 9)
    check("flagstaff opt k=6", fl["opt_k"] == 6)
    am = fl["variants"]["absmean"]
    check("absmean support optimal", am["k"] == 6 and am["support_ratio"] == 0.0)
    check("absmean scale-only trit 0", am["trit"] == 0 and am["excess"] > 0.05)
    check("flagstaff drop == absmean", fl["collapse"]["drop_eq_absmean"])
    check("flagstaff clip distinct", not fl["collapse"]["clip_eq_absmean"])
    check("no float mismatch on seed", all(rows[i]["tie_trit"] == 1 for i in
                                           ("flagstaff-mtn", "xanadu", "golden-clear-creek")))
    check("refit = opt alpha", abs(am["refit_alpha"] - fl["opt_alpha"]) < 1e-12)
    # deterministic chain
    check("deterministic", certify_seed(parse_seed(FIXTURE))["key"] == rep["key"])
    check("prev changes key", certify_seed(parse_seed(FIXTURE), "f" * 16)["key"] != rep["key"])
    # xanadu: the clip gamma (the one junior_bitnet.winsor.pack and the walk use)
    # turns -0.5 into -1 (0.5 / 0.98333 > 1/2): k = 4 vs optimum 3, support
    # excess far above RATE, so the trits themselves are off -> -1.
    xa = rows["xanadu"]["variants"]
    check("xanadu opt k=3", rows["xanadu"]["opt_k"] == 3)
    check("xanadu clip support -1", xa["clip"]["k"] == 4 and xa["clip"]["trit"] == -1)
    check("xanadu absmean scale-only 0", xa["absmean"]["trit"] == 0)
    # a seed without xanadu and the malformed rows has no -1 and exits 0
    good = parse_seed(FIXTURE)
    good["nodes"] = [good["nodes"][0], good["nodes"][2]]
    check("good rc 0", certify_seed(good)["rc"] == 0)
    return close_rule(rep["key"], pack([1, 0, -1]), "selftest:%d" % ok)


# ------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seed", help="stonefield.json (read only)")
    ap.add_argument("--matrix", help='rows ";" separated, entries "," separated')
    ap.add_argument("--prev", default=GENESIS)
    args = ap.parse_args(argv)
    if args.selftest:
        print("selftest ok key=" + selftest())
        return 0
    if len(args.prev) != 16 or any(c not in "0123456789abcdef" for c in args.prev):
        print("prev must be 16 lowercase hex chars", file=sys.stderr)
        return 2
    try:
        if args.seed:
            with open(args.seed, "r", encoding="utf-8") as fh:
                seed = parse_seed(fh.read())
        elif args.matrix:
            seed = {"covenant": {"scrape": False, "private_land": False},
                    "nodes": [{"id": "matrix", "public": True,
                               "holds_matrix": parse_matrix(args.matrix)}]}
        else:
            ap.print_help()
            return 2
    except (OSError, ValueError, ZeroDivisionError) as exc:
        print("unreadable input: %s" % exc, file=sys.stderr)
        return 2
    rep = certify_seed(seed, args.prev)
    print(json.dumps(rep, indent=1, sort_keys=True))
    return rep["rc"]


if __name__ == "__main__":
    raise SystemExit(main())
