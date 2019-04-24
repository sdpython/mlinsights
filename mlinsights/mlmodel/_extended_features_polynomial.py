"""
@file
@brief Implements new features such as polynomial features.
"""
from itertools import combinations, chain
from itertools import combinations_with_replacement as combinations_w_r


def _transform_iall(degree, bias, XP, X, multiply, final):
    "Computes the polynomial features"
    if bias:
        XP[:, 0] = 1
        pos = 1
    else:
        pos = 0

    n = X.shape[1]
    for d in range(0, degree):
        if d == 0:
            XP[:, pos:pos + n] = X
            index = list(range(pos, pos + n))
            pos += n
            index.append(pos)
        else:
            new_index = []
            end = index[-1]
            for i in range(0, n):
                a = index[i]
                new_index.append(pos)
                new_pos = pos + end - a
                multiply(XP[:, a:end], X[:, i:i + 1],
                         XP[:, pos:new_pos])
                pos = new_pos

            new_index.append(pos)
            index = new_index

    return final(XP)


def _transform_ionly(degree, bias, XP, X, multiply, final):
    "Computes the polynomial features"
    if bias:
        XP[:, 0] = 1
        pos = 1
    else:
        pos = 0

    n = X.shape[1]
    for d in range(0, degree):
        if d == 0:
            XP[:, pos:pos + n] = X
            index = list(range(pos, pos + n))
            pos += n
            index.append(pos)
        else:
            new_index = []
            end = index[-1]
            for i in range(0, n):
                a = index[i]
                new_index.append(pos)
                dec = index[i + 1] - index[i]
                new_pos = pos + end - a - dec
                if new_pos <= pos:
                    break
                multiply(XP[:, a + dec:end], X[:, i:i + 1],
                         XP[:, pos:new_pos])
                pos = new_pos

            new_index.append(pos)
            index = new_index

    return final(XP)


def _combinations_poly(n_features, degree, interaction_only, include_bias):
    "Computes all polynomial features combinations."
    comb = (combinations if interaction_only else combinations_w_r)
    start = int(not include_bias)
    return chain.from_iterable(comb(range(n_features), i)
                               for i in range(start, degree + 1))
