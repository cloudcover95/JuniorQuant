"""I2_S unpack next to src/trit_series.py. Not ATML. Not SVD."""
from __future__ import annotations

from src.trit_series import center, delta


def pack_i2s(trits) -> tuple[int, int]:
    bits = 0
    n = 0
    for t in trits:
        tt = 1 if t > 0 else (-1 if t < 0 else 0)
        bits = (bits << 2) | (tt + 1)
        n += 1
    return bits, n


def unpack_i2s(bits: int, n: int) -> list[int]:
    if n <= 0:
        return []
    out = [0] * n
    tmp = int(bits)
    for i in range(n - 1, -1, -1):
        out[i] = (tmp & 3) - 1
        tmp >>= 2
    return out


def series_trits(xs, mode: str = "center") -> dict:
    walked = center(xs) if mode == "center" else delta(xs)
    trits = []
    for x in walked or [0.0]:
        if x > 0:
            trits.append(1)
        elif x < 0:
            trits.append(-1)
        else:
            trits.append(0)
    bits, n = pack_i2s(trits)
    return {"mode": mode, "trits": trits, "i2s_hex": format(bits, "x"), "n": n}
