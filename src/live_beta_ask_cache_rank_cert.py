"""Certified top-1 for a ternary retrieval index, so juniorctl ask cache hits stay sound.

Stdlib only, offline. Ports touched: Home, OSai, FieldCore, Flagstaff, JuniorLLM.

Roadmap item: JuniorLLM docs/BETA_TO_OS.md "Live beta", steps 1-3 as named by
Quant Chief for pair A (slot 2026-10-18 05:20 MT): `juniorctl health`, `ask`,
`night`, and `tests/test_junior_aie.py`. This file is the quant-error /
Winsor-bound side of `juniorctl ask` and test_junior_aie `test_ask_and_cache`
only. It does not touch `security`, `test_linux_security`, backfill_check or
stonefield. The I2_S key side belongs to Trit Pack & I2_S.

The gap: `juniorctl ask` ranks retrieval documents and test_ask_and_cache
requires the second identical ask to come back `cached`. If the document side
of the index is stored ternary (BitNet b1.58 style, d_hat = gamma * t,
t = RoundClip(d / gamma, -1, 1), JuniorHome docs/QUANT_ERROR.md), a cached
answer is only sound when the float index would have picked the same top-1
document. Earlier pair-A files gate run counts (SPRT), drift and outliers; none
says when a single ternary ranking, and therefore a cache entry, is provably
the float answer.

New capability: a per-query certificate that needs only two stored floats per
row. Split each float row along its ternary code: d = alpha_d * d_hat + r_d with
alpha_d = <d, d_hat> / <d_hat, d_hat> (least-squares scale) and r_d orthogonal
to d_hat; store alpha_d and rho_d = ||r_d||. Then exactly
  s_d = <q, d> = alpha_d * s_hat_d + <q, r_d>,   |<q, r_d>| <= ||q|| * rho_d,
so the ternary winner a = argmax s_hat (the answer the cache stores) is also
the float winner whenever
  alpha_a s_hat_a - ||q|| rho_a > alpha_j s_hat_j + ||q|| rho_j + fp_slack
for every j != a. (A plain ||q|| * ||e_a - e_j|| bound on e = d - d_hat is far
too loose, since most of e is parallel to d_hat.) The Winsor gamma decides which
coordinates survive the 1/2 threshold, so it sets rho_d.
A query that passes is "cache-sound": caching its ternary answer can never
disagree with a float recompute, under any of the three Winsor gammas
(JuniorHome docs/WINSOR.md):
  absmean = mean|W|                                   (BitNet b1.58)
  drop    = mean|W| after dropping the top 1/16       (Junior drop)
  p95     = mean of min(|W|, P95), nearest-rank       (P95 clip)
A row whose code is all zero (for example Junior drop on a row with at most
n/16 nonzeros, where gamma = 0) has rho = ||d|| and is reported as dead_rows.
gamma is per document row. If a query is cache-sound under every gamma, all
three ternary indexes return the same document (the float one), so a cache
filled by one host is valid on another host using a different gamma.

Verdict trit per gamma, u = uncertified fraction of queries:
  +1 u == 0, 0 0 < u <= RATE, -1 u > RATE (do not cache uncertified asks).

Close rule: sha256(prev || pack || text)[:16], GENESIS prev = "0" * 16. Per
gamma: pack = I2_S 2-bit code (0->0, 1->1, -1->2, MSB first) of the per-query
cache-sound trits (+1 sound, -1 not), text = canonical JSON of that gamma's
summary. Chained absmean -> drop -> p95, then report key: pack = I2_S code of
the 3 verdicts, text = canonical JSON of the report. Rate 0.05.
Compatible engines: BitNet-mlx, JuniorLLM, JuniorQuant, bitnetd 127.0.0.1:8765,
i2sd 127.0.0.1:8767 (named, never dialed), JuniorOS overlay (Debian/Alpine).
Bind 127.0.0.1 only. mint/socket/layer_one/fetch/ue5_launch false.
No model pull, no GGUF download, no network, no team-module import.

Usage:
  python3 live_beta_ask_cache_rank_cert.py --selftest
  python3 live_beta_ask_cache_rank_cert.py --synthetic 200 --queries 300 --seed 7
  python3 live_beta_ask_cache_rank_cert.py --ask "red feather granite"
"""
from __future__ import annotations

import argparse
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
BITNETD_PORT = 8765
I2SD_PORT = 8767
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
GENESIS = "0" * 16
U = 2.0 ** -53
GAMMAS = ("absmean", "drop", "p95")
DIM = 256
TAPS = 4
_ENC = {0: 0, 1: 1, -1: 2}

SEED_DOCS = (
    "red feather lakes granite boulders",
    "flagstaff basalt crag morning shade",
    "fieldcore sensor night drift log",
    "home edge bench ternary absmean",
    "covenant private land boulders owner consent",
)
WORDS = (
    "red feather lakes granite boulders flagstaff basalt crag morning shade "
    "fieldcore sensor night drift log home edge bench ternary absmean covenant "
    "private land owner consent trail approach roof slab crimp sloper dyno "
    "highball pad spot chalk brush rain dry wind cold warm north south east west "
    "creek canyon ridge meadow aspen pine juniper sandstone quartz gneiss"
).split()


def pack_i2s(trits) -> bytes:
    acc = bits = 0
    out = bytearray()
    for t in trits:
        acc = (acc << 2) | _ENC[t]
        bits += 2
        if bits == 8:
            out.append(acc)
            acc = bits = 0
    if bits:
        out.append(acc << (8 - bits))
    return bytes(out)


def close_rule(prev: str, pack: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode() + pack + text.encode()).hexdigest()[:16]


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def embed(text: str, dim: int = DIM):
    """Hashed bag of words, TAPS signed taps per token, L2 normalised."""
    v = [0.0] * dim
    for tok in text.lower().split():
        h = hashlib.sha256(tok.encode()).digest()
        for k in range(TAPS):
            idx = int.from_bytes(h[3 * k:3 * k + 2], "big") % dim
            sign = 1.0 if h[3 * k + 2] & 1 else -1.0
            mag = 0.5 + (h[3 * k + 2] >> 1) / 254.0
            v[idx] += sign * mag
    n = math.sqrt(math.fsum(x * x for x in v))
    return [x / n for x in v] if n > 0 else v


def gamma(w, which: str) -> float:
    a = sorted(abs(x) for x in w)
    n = len(a)
    if which == "absmean":
        return math.fsum(a) / n
    if which == "drop":
        d = n // 16
        t = a[: n - d] if d else a
        return math.fsum(t) / len(t)
    p = a[max(0, math.ceil(0.95 * n) - 1)]
    return math.fsum(min(x, p) for x in a) / n


def quantize(w, g: float):
    out = []
    for x in w:
        r = abs(x) / g if g > 0 else 0.0
        out.append(0 if r < 0.5 else (1 if x > 0 else -1))
    return out


def dot(a, b) -> float:
    return math.fsum(x * y for x, y in zip(a, b))


def norm(a) -> float:
    return math.sqrt(math.fsum(x * x for x in a))


def build_index(docs, which: str):
    rows = []
    for text in docs:
        d = embed(text)
        g = gamma(d, which)
        t = quantize(d, g)
        dh = [g * x for x in t]
        hh = dot(dh, dh)
        alpha = dot(d, dh) / hh if hh > 0 else 0.0
        r = [x - alpha * y for x, y in zip(d, dh)]
        rows.append({"d": d, "dh": dh, "e": [x - y for x, y in zip(d, dh)],
                     "g": g, "t": t, "alpha": alpha, "rho": norm(r),
                     "nd": norm(d), "ndh": norm(dh)})
    return rows


def fp_slack(nq: float, ra: dict, rj: dict, dim: int = DIM) -> float:
    # Generous rounding allowance for the dots, alpha and rho (all O(dim * U)).
    big = nq * (ra["nd"] + rj["nd"] + abs(ra["alpha"]) * ra["ndh"] + abs(rj["alpha"]) * rj["ndh"])
    return 16 * dim * U * big + 1e-300


def rank(rows, q, key: str):
    s = [dot(q, r[key]) for r in rows]
    best = max(range(len(s)), key=lambda i: (s[i], -i))
    return best, s


def certify(rows, q) -> dict:
    nq = norm(q)
    a, sh = rank(rows, q, "dh")
    ra = rows[a]
    low = ra["alpha"] * sh[a] - nq * ra["rho"]
    worst = math.inf
    sound = True
    for j, rj in enumerate(rows):
        if j == a:
            continue
        up = rj["alpha"] * sh[j] + nq * rj["rho"]
        slack = low - up - fp_slack(nq, ra, rj)
        worst = min(worst, slack)
        if slack <= 0:
            sound = False
    return {"top": a, "sound": sound, "worst_slack": worst}


def evaluate(docs, queries) -> dict:
    key = GENESIS
    report = {"rate": RATE, "dim": DIM, "docs": len(docs), "queries": len(queries),
              "bind": BIND, "ports": list(PORTS), "gammas": {}}
    tops = {}
    verdicts = []
    for which in GAMMAS:
        rows = build_index(docs, which)
        res = [certify(rows, embed(q)) for q in queries]
        bad = sum(1 for r in res if not r["sound"])
        u = bad / len(res) if res else 0.0
        v = 1 if bad == 0 else (0 if u <= RATE else -1)
        e2 = math.fsum(x * x for r in rows for x in r["e"])
        s2 = math.fsum(x * x for r in rows for x in r["d"])
        summ = {"gamma": which, "uncertified": bad, "u": u, "verdict": v,
                "rel_sse": e2 / s2 if s2 else 0.0,
                "min_slack": float.hex(min(r["worst_slack"] for r in res)) if res else None,
                "max_rho": float.hex(max(r["rho"] for r in rows)),
                "dead_rows": sum(1 for r in rows if not any(r["t"]))}
        key = close_rule(key, pack_i2s([1 if r["sound"] else -1 for r in res]), canon(summ))
        summ["key"] = key
        report["gammas"][which] = summ
        tops[which] = res
        verdicts.append(v)
    agree = 0
    for i in range(len(queries)):
        rs = [tops[w][i] for w in GAMMAS]
        if all(r["sound"] for r in rs) and len({r["top"] for r in rs}) == 1:
            agree += 1
    report["cross_gamma_cache_sound"] = agree
    report["verdicts"] = verdicts
    report["key"] = close_rule(key, pack_i2s(verdicts), canon(report))
    return report, tops


def synthetic(n_docs: int, n_q: int, seed: int):
    rng = random.Random(seed)
    docs = list(SEED_DOCS)
    while len(docs) < n_docs:
        docs.append(" ".join(rng.sample(WORDS, rng.randint(4, 7))))
    qs = []
    for i in range(n_q):
        base = docs[rng.randrange(len(docs))].split()
        k = max(2, len(base) - rng.randint(1, 3))
        q = rng.sample(base, k)
        if i % 5 == 0:
            q.append(rng.choice(WORDS))
        qs.append(" ".join(q))
    return docs, qs


def selftest() -> int:
    fails = []

    def ok(name, cond):
        print(("ok   " if cond else "FAIL ") + name)
        if not cond:
            fails.append(name)

    ok("flags false, bind loopback", not (MINT or SOCKET or LAYER_ONE or FETCH or UE5_LAUNCH)
       and BIND == "127.0.0.1" and I2SD_PORT == 8767 and BITNETD_PORT == 8765)
    ok("pack_i2s codes", pack_i2s([1, -1, 0, 1]) == bytes([0b01100001])
       and pack_i2s([-1]) == bytes([0b10000000]))
    k1 = close_rule(GENESIS, b"\x01", "x")
    ok("close rule 16 hex, pack-sensitive",
       len(k1) == 16 and k1 == hashlib.sha256(b"0" * 16 + b"\x01x").hexdigest()[:16]
       and k1 != close_rule(GENESIS, b"\x02", "x"))
    v = embed("red feather lakes granite boulders")
    ok("gamma order drop<=absmean, p95<=absmean",
       gamma(v, "drop") <= gamma(v, "absmean") + 1e-15
       and gamma(v, "p95") <= gamma(v, "absmean") + 1e-15)

    docs, qs = synthetic(200, 300, 7)
    sound_all = True
    bound_all = True
    for which in GAMMAS:
        rows = build_index(docs, which)
        for q in qs[:120]:
            qe = embed(q)
            c = certify(rows, qe)
            ft, fs = rank(rows, qe, "d")
            _, hs = rank(rows, qe, "dh")
            if c["sound"] and ft != c["top"]:
                sound_all = False
            nq = norm(qe)
            for j in range(0, len(rows), 7):
                rj = rows[j]
                lhs = abs(fs[j] - rj["alpha"] * hs[j])
                if lhs > nq * rj["rho"] + fp_slack(nq, rj, rj):
                    bound_all = False
    ok("certified => float top-1 equals ternary top-1", sound_all)
    ok("|s_d - alpha_d s_hat_d| <= ||q|| rho_d holds", bound_all)

    fx = ["red feather lakes granite boulders"] + list(SEED_DOCS[1:])
    rep, tops = evaluate(fx, ["red feather granite"])
    ok("test_ask_and_cache fixture cache-sound under all gammas",
       all(tops[w][0]["sound"] and tops[w][0]["top"] == 0 for w in GAMMAS)
       and rep["cross_gamma_cache_sound"] == 1 and rep["verdicts"] == [1, 1, 1])

    dup = ["granite boulders red", "granite boulders red", "basalt crag shade"]
    rep2, tops2 = evaluate(dup, ["granite boulders red"] * 3 + ["basalt crag shade"])
    ok("exact tie is never certified, verdict -1 above rate",
       not tops2["absmean"][0]["sound"] and tops2["absmean"][3]["sound"]
       and rep2["verdicts"] == [-1, -1, -1])

    r1, _ = evaluate(docs[:60], qs[:80])
    r2, _ = evaluate(docs[:60], qs[:80])
    ok("report key deterministic", r1["key"] == r2["key"] and len(r1["key"]) == 16)
    ok("verdicts are trits", all(x in (-1, 0, 1) for x in r1["verdicts"]))
    print("selftest", "PASS" if not fails else "FAIL", r1["key"])
    return 0 if not fails else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ask", type=str, default=None)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.ask is not None:
        rep, tops = evaluate(list(SEED_DOCS), [a.ask])
        rep["answer"] = {w: SEED_DOCS[tops[w][0]["top"]] for w in GAMMAS}
        rep["cache_sound"] = {w: tops[w][0]["sound"] for w in GAMMAS}
        print(json.dumps(rep, indent=2, sort_keys=True))
        return 0
    docs, qs = synthetic(max(a.synthetic, len(SEED_DOCS)), a.queries, a.seed)
    rep, _ = evaluate(docs, qs)
    print(json.dumps(rep, indent=2, sort_keys=True))
    return 0 if min(rep["verdicts"]) >= 0 else 2


if __name__ == "__main__":
    sys.exit(main())
