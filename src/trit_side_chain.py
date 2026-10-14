"""Side-commit chain verifier next to src/trit_side_commit.py. Pure math, no I/O."""
from __future__ import annotations

from src.trit_side_commit import LAYER_ONE, MINT, RATE, SOCKET, pack_bytes, side_commit

GENESIS = "0" * 16


def build_chain(rows, prev: str = GENESIS) -> list[dict]:
    """rows: iterable of (trits, text). Each link = sha256(prev || pack || text)[:16]."""
    out = []
    for trits, text in rows:
        c = side_commit(prev, pack_bytes(trits), text)
        out.append({"prev": prev, "trits": list(trits), "text": text, "commit": c})
        prev = c
    return out


def verify_chain(chain, prev: str = GENESIS) -> dict:
    for i, link in enumerate(chain):
        if link["prev"] != prev:
            return {"ok": False, "at": i, "reason": "prev"}
        if side_commit(prev, pack_bytes(link["trits"]), link["text"]) != link["commit"]:
            return {"ok": False, "at": i, "reason": "commit"}
        prev = link["commit"]
    return {"ok": True, "n": len(chain), "tip": prev}


def rate_accumulate(chain) -> float:
    """Rate-weighted trit sum: RATE * sum(trits) over every link."""
    return RATE * sum(sum(link["trits"]) for link in chain)


if __name__ == "__main__":
    ch = build_chain([([1, 0, -1], "a"), ([1, 1, 0], "b"), ([-1, -1, 1], "c")])
    v = verify_chain(ch)
    assert v["ok"] and v["n"] == 3
    bad = [dict(link) for link in ch]
    bad[1]["text"] = "x"
    assert verify_chain(bad) == {"ok": False, "at": 1, "reason": "commit"}
    assert abs(rate_accumulate(ch) - 0.05) < 1e-12
    assert not (MINT or SOCKET or LAYER_ONE)
    print({"tip": v["tip"], "rate_sum": rate_accumulate(ch), "ok": True})
