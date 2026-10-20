"""BETA_TO_OS Live beta step 5 ("backfill_check.py exits 0"): substance gate.

Stdlib only, offline, read-only. Ports touched: Home, OSai, FieldCore,
Flagstaff, JuniorLLM. TCP ports named, never dialed: bitnetd 127.0.0.1:8765,
i2sd userspace 127.0.0.1:8767. Bind 127.0.0.1 only; this file opens no socket.
mint false, socket false, layer_one false, fetch false, ue5_launch false.
No model pull, no GGUF download, no subprocess, no network, never imports or
runs backfill_check.py (EXPECTED is read from its source with ast).

Roadmap item: JuniorLLM docs/BETA_TO_OS.md "Live beta", step 5
("`python scripts/backfill_check.py` exits 0", on "A fresh clone");
docs/COMPILED_BACKLOG.md section A (live beta suite). Quant Chief, pair A,
2026-10-19 18:20 MT run: step 5, real gap only. Quant-error / Winsor side.

The gap: backfill_check.py tests `(ROOT / p).is_file()`, so its exit code is a
1-bit quantization of "the compiled backlog skeleton exists". A zero-byte or
whitespace-only placeholder (an empty grok_bot/LAST_RECEIPT.md, a truncated
STATE.md) still exits 0. The exit-code trits (BitNet-mlx 9244444), the SPRT
run gate (JuniorQuant adfb204), the coverage handshake (829a789) and the
fresh-clone index trits (JuniorLLM scripts/backfill_fresh_clone_index_i2s_key.py)
all take presence or blob identity as given; none bounds how much substance
the 12 files carry, so a hollow skeleton passes step 5 everywhere.

New capability (a continuous substance measure, then a certified ternary
quantization of it, with its quant error):
  1. s_i = non-whitespace byte count of EXPECTED path i (0 when missing or not
     a regular file). Whitespace (space, tab, CR, LF, FF, VT) is excluded, so
     s_i is invariant under core.autocrlf and trailing-newline edits: the same
     verdict on a Linux, macOS or Windows fresh clone.
  2. x_i = log2(1 + s_i); d_i = x_i - lower median(x). Winsor gammas
     (JuniorHome docs/WINSOR.md, engine-matched to JuniorLLM junior_bitnet):
       absmean  gamma = mean|d|                      (junior_bitnet.math.absmean)
       drop     gamma = mean|d| after dropping the top n // 16 (== absmean, n < 16)
       clip     tau = sorted|d|[floor(0.95 (n - 1))], gamma = mean min(|d|, tau),
                gamma_eff = gamma + 1e-7             (junior_bitnet.winsor.pack)
     Per-variant BitNet trit q_i = clip(round(d_i / gamma_eff), -1, 1).
  3. Thin bound at level RATE. mean|d| about the median is the maximum
     likelihood Laplace scale b, and P(d < -k b) = exp(-k) / 2, so the one-sided
     RATE tail sits at k = ln(1 / (2 RATE)) = ln 10 = 2.302585 for RATE 0.05.
     With the RATE slack on the scale, gamma_hi = gamma_eff (1 + RATE), a path is
     certified thin under a variant when d_i < -k gamma_hi.
  4. Path trit p_i:
       -1 hollow  s_i == 0 (missing, not a file, empty or whitespace-only)
        0 thin    s_i > 0 and certified thin under all three variants
       +1 substantive otherwise
     The thin test needs n >= 4 positive files; below that it is skipped.
  5. Quant error per variant (JuniorHome docs/QUANT_ERROR.md):
     SSE = sum (d - gamma q)^2, rel = SSE / sum d^2, zero-band share, and the
     RATE stability count: paths whose q flips at gamma (1 -+ 0.05).
  Verdict trit v: -1 if any hollow, 0 if any thin, else +1. Exit codes 0 / 2 / 1
  for +1 / 0 / -1 (step 5's "exits 0" holds with substance only when v = +1).

Rows: one per path [q_absmean, q_drop, q_clip, p], one per variant
[sign(rel <= 0.5), stable, v], then the verdict [v, thin_tested, all_present].
I2_S pack: codes 0->0, 1->1, -1->2, four per byte MSB first, zero padded.
Close rule: key = sha256(prev || pack || text)[:16], GENESIS prev = "0" * 16
as ASCII hex, text = canonical JSON of the row (sorted keys, compact), chained
row to row. Rate lock: sha256(prev || pack || b"0.05").hexdigest() at the end.

It does not wrap, test, or re-derive any team file or earlier Winsor file
(JuniorQuant src/winsor_clip.py, src/i2sd_*, src/live_beta_*; JuniorStock
risk/winsor_quant_error.py, security/*, quant/calibration/*.py), and it never
reads .git (the index side belongs to Trit Pack & I2_S).

Run: python3 src/live_beta_step5_backfill_substance_gate.py --selftest
     python3 src/live_beta_step5_backfill_substance_gate.py --root ../JuniorLLM [--prev HEX16]
Compatible engines: JuniorQuant, JuniorStock, JuniorLLM junior_bitnet
(absmean, winsor.pack), BitNet-mlx, bitnetd, i2sd (same I2_S code map and
close rule).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import sys
import tempfile
from math import floor
from pathlib import Path

RATE = 0.05
GENESIS = "0" * 16
ENGINE_EPS = 1e-7
K_THIN = math.log(1.0 / (2.0 * RATE))
MIN_THIN_N = 4
VARIANTS = ("absmean", "drop", "clip")
PORTS = ("Home", "OSai", "FieldCore", "Flagstaff", "JuniorLLM")
TCP = {"bitnetd": "127.0.0.1:8765", "i2sd": "127.0.0.1:8767"}
FLAGS = {"mint": False, "socket": False, "layer_one": False, "fetch": False, "ue5_launch": False}
ENGINES = ("JuniorQuant", "JuniorStock", "JuniorLLM junior_bitnet", "BitNet-mlx", "bitnetd", "i2sd")
ROADMAP_ITEM = "JuniorLLM docs/BETA_TO_OS.md Live beta step 5 (COMPILED_BACKLOG A)"
BACKFILL = "scripts/backfill_check.py"
CODE = {0: 0, 1: 1, -1: 2}
WS = b" \t\r\n\x0b\x0c"
EXIT = {1: 0, 0: 2, -1: 1}


# ---------------------------------------------------------------- pack / close
def pack(trits: list[int]) -> bytes:
    out = bytearray()
    for i in range(0, len(trits), 4):
        chunk = list(trits[i:i + 4]) + [0] * (4 - len(trits[i:i + 4]))
        b = 0
        for t in chunk:
            b = (b << 2) | CODE[t]
        out.append(b)
    return bytes(out)


def close_rule(prev: str, pk: bytes, text: str) -> str:
    return hashlib.sha256(prev.encode() + pk + text.encode()).hexdigest()[:16]


def rate_lock(prev: str, pk: bytes) -> str:
    return hashlib.sha256(prev.encode() + pk + str(RATE).encode()).hexdigest()


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


# ---------------------------------------------------------------- EXPECTED
def read_expected(src: str) -> list[str]:
    """Literal EXPECTED list from backfill_check.py source; never imported."""
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "EXPECTED":
                    val = ast.literal_eval(node.value)
                    if not isinstance(val, (list, tuple)) or not all(isinstance(p, str) for p in val):
                        raise ValueError("EXPECTED is not a literal list of str")
                    return list(val)
    raise ValueError("no EXPECTED assignment in backfill_check.py")


def substance(path: Path) -> int:
    if path.is_symlink() or not path.is_file():
        return 0
    return len(path.read_bytes().translate(None, WS))


# ---------------------------------------------------------------- winsor gammas
def clip_round(x: float) -> int:
    return max(-1, min(1, int(round(x))))


def lower_median(xs: list[float]) -> float:
    s = sorted(xs)
    return s[(len(s) - 1) // 2]


def gammas(d: list[float]) -> dict:
    a = [abs(x) for x in d]
    n = len(a)
    if n == 0:
        return {"absmean": 1.0, "drop": 1.0, "clip": 1.0, "tau": 0.0}
    absmean = sum(a) / n or 1.0
    k = n // 16
    kept = sorted(a)[: n - k]
    drop = sum(kept) / len(kept) or 1.0
    s = sorted(a)
    tau = s[min(n - 1, max(0, floor(0.95 * (n - 1))))]
    clip = sum(min(x, tau) for x in a) / n or 1.0
    return {"absmean": absmean, "drop": drop, "clip": clip, "tau": tau}


def eff(variant: str, g: float) -> float:
    return g + ENGINE_EPS if variant == "clip" else g


def quant_error(d: list[float], g: float) -> dict:
    t = [clip_round(x / g) for x in d]
    sse = sum((x - g * ti) ** 2 for x, ti in zip(d, t))
    en = sum(x * x for x in d)
    return {"sse": round(sse, 9), "rel": round(sse / en, 9) if en else 0.0,
            "zero_share": round(t.count(0) / len(t), 9) if t else 0.0}


def flips(d: list[float], g: float) -> int:
    base = [clip_round(x / g) for x in d]
    n = 0
    for f in (1.0 - RATE, 1.0 + RATE):
        alt = [clip_round(x / (g * f)) for x in d]
        n += sum(1 for a, b in zip(base, alt) if a != b)
    return n


# ---------------------------------------------------------------- gate
def gate(sizes: dict[str, int], prev: str = GENESIS) -> dict:
    paths = list(sizes)
    pos = [p for p in paths if sizes[p] > 0]
    thin_tested = len(pos) >= MIN_THIN_N
    x = {p: math.log2(1 + sizes[p]) for p in pos}
    m = lower_median(list(x.values())) if pos else 0.0
    d = {p: x[p] - m for p in pos}
    dl = [d[p] for p in pos]
    g = gammas(dl)
    geff = {v: eff(v, g[v]) for v in VARIANTS}

    rows, key, all_trits = [], prev, []
    path_out = []
    for p in paths:
        if sizes[p] == 0:
            q = {v: -1 for v in VARIANTS}
            pt, thin = -1, {v: False for v in VARIANTS}
        else:
            q = {v: clip_round(d[p] / geff[v]) for v in VARIANTS}
            thin = {v: thin_tested and d[p] < -K_THIN * geff[v] * (1.0 + RATE) for v in VARIANTS}
            pt = 0 if all(thin.values()) else 1
        row = {"kind": "path", "path": p, "s": sizes[p],
               "d": round(d.get(p, 0.0), 9), "q": q, "thin": thin, "trit": pt}
        trits = [q["absmean"], q["drop"], q["clip"], pt]
        pk = pack(trits)
        key = close_rule(key, pk, canon(row))
        row["key"] = key
        rows.append(row)
        all_trits += trits
        path_out.append(pt)

    v = -1 if -1 in path_out else (0 if 0 in path_out else 1)
    var_out = {}
    for name in VARIANTS:
        qe = quant_error(dl, geff[name]) if dl else {"sse": 0.0, "rel": 0.0, "zero_share": 0.0}
        fl = flips(dl, geff[name]) if dl else 0
        row = {"kind": "variant", "variant": name, "gamma": round(g[name], 9),
               "gamma_eff": round(geff[name], 9),
               "thin_bound": round(-K_THIN * geff[name] * (1.0 + RATE), 9),
               "flips_rate": fl, **qe}
        trits = [1 if qe["rel"] <= 0.5 else -1, 1 if fl == 0 else 0, v]
        pk = pack(trits)
        key = close_rule(key, pk, canon(row))
        row["key"] = key
        rows.append(row)
        all_trits += trits
        var_out[name] = row

    all_present = all(sizes[p] > 0 for p in paths)
    vrow = {"kind": "verdict", "verdict": v, "exit": EXIT[v], "n": len(paths),
            "hollow": [p for p, t in zip(paths, path_out) if t == -1],
            "thin": [p for p, t in zip(paths, path_out) if t == 0],
            "thin_tested": thin_tested, "k_thin": round(K_THIN, 9),
            "median_log2": round(m, 9), "tau": round(g["tau"], 9),
            "rate": RATE, "roadmap": ROADMAP_ITEM}
    trits = [v, 1 if thin_tested else 0, 1 if all_present else -1]
    pk = pack(trits)
    key = close_rule(key, pk, canon(vrow))
    vrow["key"] = key
    rows.append(vrow)
    all_trits += trits
    full = pack(all_trits)
    return {"rows": rows, "verdict": v, "exit": EXIT[v], "key": key,
            "rate_lock": rate_lock(key, full), "pack_hex": full.hex(),
            "ports": list(PORTS), "tcp": TCP, "flags": FLAGS, "engines": list(ENGINES)}


def scan(root: Path) -> dict[str, int]:
    src = (root / BACKFILL).read_text(encoding="utf-8")
    return {p: substance(root / p) for p in read_expected(src)}


# ---------------------------------------------------------------- selftest
SEED_SRC = '''EXPECTED = [
    "docs/A.md", "docs/B.md", "STATE.md", "evals/c.py", "rails/d.sh",
    "ports/E.md", "grok_bot/LAST_RECEIPT.md", "x/f.py",
]
'''
SEED_BODY = {"docs/A.md": 3820, "docs/B.md": 1045, "STATE.md": 333, "evals/c.py": 335,
             "rails/d.sh": 2847, "ports/E.md": 526, "grok_bot/LAST_RECEIPT.md": 1114,
             "x/f.py": 4067}


def _write_tree(root: Path, body: dict[str, bytes]) -> None:
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / BACKFILL).write_text(SEED_SRC, encoding="utf-8")
    for p, data in body.items():
        f = root / p
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(data)


def selftest() -> int:
    ok = True

    def check(cond: bool, msg: str) -> None:
        nonlocal ok
        if not cond:
            ok = False
            print("FAIL", msg)

    # pack / close conventions
    check(pack([1, -1, 0, 1]) == bytes([0b01100001]), "pack MSB-first code map")
    check(pack([-1]) == bytes([0b10000000]), "pack zero padding")
    check(close_rule(GENESIS, b"", "") == hashlib.sha256(GENESIS.encode()).hexdigest()[:16], "close rule")
    check(abs(K_THIN - math.log(10.0)) < 1e-12, "k_thin = ln 10 at RATE 0.05")
    check(read_expected(SEED_SRC) == list(SEED_BODY), "ast EXPECTED read")
    try:
        read_expected("EXPECTED = [p for p in 'ab']\n")
        check(False, "non-literal EXPECTED rejected")
    except ValueError:
        pass
    # gammas: drop == absmean when n < 16, clip <= absmean
    g = gammas([0.5, -1.0, 2.0, -0.25])
    check(g["drop"] == g["absmean"], "drop == absmean for n < 16")
    check(g["clip"] <= g["absmean"] + 1e-15, "clip <= absmean")
    g16 = gammas([float(i) for i in range(16)] + [100.0])
    check(g16["drop"] < g16["absmean"], "drop removes top 1/16 for n >= 16")
    # Laplace tail sanity: exp(-k)/2 == RATE
    check(abs(math.exp(-K_THIN) / 2 - RATE) < 1e-12, "laplace one-sided tail == RATE")

    # 1) substantive tree (sizes from the JuniorLLM tree measured 2026-10-19 18:20 MT)
    with tempfile.TemporaryDirectory() as td:
        r = Path(td)
        _write_tree(r, {p: (b"x" * n) + b"\r\n" for p, n in SEED_BODY.items()})
        a = gate(scan(r))
        check(a["verdict"] == 1 and a["exit"] == 0, f"substantive tree passes {a['verdict']}")
        # CRLF / whitespace invariance
        _write_tree(r, {p: b"  " + (b"x" * n) + b"\n\n\t" for p, n in SEED_BODY.items()})
        b = gate(scan(r))
        check(b["key"] == a["key"], "whitespace/CRLF invariant key")
        # determinism
        check(gate(scan(r))["key"] == b["key"], "deterministic key")
        # prev changes the chain
        check(gate(scan(r), prev="f" * 16)["key"] != b["key"], "prev feeds the chain")

    # 2) whitespace-only receipt: backfill_check exits 0, gate says hollow
    with tempfile.TemporaryDirectory() as td:
        r = Path(td)
        body = {p: b"x" * n for p, n in SEED_BODY.items()}
        body["grok_bot/LAST_RECEIPT.md"] = b" \n\r\n\t\n"
        _write_tree(r, body)
        present = all((r / p).is_file() for p in SEED_BODY)
        c = gate(scan(r))
        check(present, "backfill_check would exit 0 (all is_file)")
        check(c["verdict"] == -1 and c["exit"] == 1, "whitespace-only file is hollow")
        check(c["rows"][-1]["hollow"] == ["grok_bot/LAST_RECEIPT.md"], "hollow path named")

    # 3) truncated stub: present, non-empty, certified thin under all gammas
    with tempfile.TemporaryDirectory() as td:
        r = Path(td)
        body = {p: b"x" * n for p, n in SEED_BODY.items()}
        body["STATE.md"] = b"# TODO"
        _write_tree(r, body)
        e = gate(scan(r))
        check(e["verdict"] == 0 and e["exit"] == 2, f"stub is thin {e['rows'][-1]}")
        check(e["rows"][-1]["thin"] == ["STATE.md"], "thin path named")

    # 4) missing file is hollow; symlink counts as 0
    with tempfile.TemporaryDirectory() as td:
        r = Path(td)
        body = {p: b"x" * n for p, n in SEED_BODY.items()}
        del body["x/f.py"]
        _write_tree(r, body)
        f = gate(scan(r))
        check(f["verdict"] == -1 and "x/f.py" in f["rows"][-1]["hollow"], "missing is hollow")
        (r / "x").mkdir(exist_ok=True)
        (r / "x/f.py").symlink_to(r / "docs/A.md")
        check(gate(scan(r))["verdict"] == -1, "symlink is not a fresh-clone regular file")

    # 5) small n: thin test skipped, only hollow can fail
    s = gate({"a": 3000, "b": 4})
    check(s["rows"][-1]["thin_tested"] is False and s["verdict"] == 1, "thin skipped for n < 4")

    # flags / bind
    check(not any(FLAGS.values()), "all flags false")
    check(all(v.startswith("127.0.0.1:") for v in TCP.values()), "loopback only")

    with tempfile.TemporaryDirectory() as td:
        r = Path(td)
        _write_tree(r, {p: b"x" * n for p, n in SEED_BODY.items()})
        key = gate(scan(r))["key"]
    print(canon({"selftest": "pass" if ok else "fail", "key": key, "rate": RATE,
                 "ports": list(PORTS), "engines": list(ENGINES)}))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--root", type=Path, help="JuniorLLM checkout (fresh clone)")
    ap.add_argument("--prev", default=GENESIS)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.root is None:
        ap.error("--root or --selftest required")
    if len(a.prev) != 16 or any(c not in "0123456789abcdef" for c in a.prev):
        ap.error("--prev must be 16 lowercase hex")
    out = gate(scan(a.root.resolve()), prev=a.prev)
    print(json.dumps(out, sort_keys=True, indent=1))
    return out["exit"]


if __name__ == "__main__":
    sys.exit(main())
