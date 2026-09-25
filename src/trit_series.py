"""T0 series maps. Not ATML. Not SVD."""

def center(xs):
    if not xs:
        return []
    mu = sum(xs) / float(len(xs))
    return [x - mu for x in xs]


def delta(xs):
    if not xs:
        return []
    out = [0.0]
    for i in range(1, len(xs)):
        out.append(xs[i] - xs[i - 1])
    return out
