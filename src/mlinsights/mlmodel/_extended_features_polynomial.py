"""
@file
@brief Implements new features such as polynomial features.
"""


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
                XP[:, pos:new_pos] = multiply(
                    XP[:, a:end], X[:, i:i + 1])
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
                new_pos = pos + end - a - d
                if new_pos <= pos:
                    break
                XP[:, pos:new_pos] = multiply(
                    XP[:, a + d:end], X[:, i:i + 1])
                pos = new_pos

            new_index.append(pos)
            index = new_index

    return final(XP)


def _transform_iall_transpose(degree, bias, XP, X, multiply, final):
    "Computes the polynomial features, matrix is transposed"
    if bias:
        XP[0, :] = 1
        pos = 1
    else:
        pos = 0

    n = X.shape[1]
    for d in range(0, degree):
        if d == 0:
            XP[pos:pos + n, :] = X.T
            X = XP[pos:pos + n, :]
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
                XP[pos:new_pos, :] = multiply(
                    XP[a:end, :], X[i:i + 1, :])
                pos = new_pos

            new_index.append(pos)
            index = new_index

    return final(XP)


def _transform_ionly_transpose(degree, bias, XP, X, multiply, final):
    "Computes the polynomial features, matrix is transposed"
    if bias:
        XP[0, :] = 1
        pos = 1
    else:
        pos = 0

    n = X.shape[1]
    for d in range(0, degree):
        if d == 0:
            XP[pos:pos + n, :] = X.T
            X = XP[pos:pos + n, :]
            index = list(range(pos, pos + n))
            pos += n
            index.append(pos)
        else:
            new_index = []
            end = index[-1]
            for i in range(0, n):
                a = index[i]
                new_index.append(pos)
                new_pos = pos + end - a - d
                if new_pos <= pos:
                    break
                XP[pos:new_pos, :] = multiply(
                    XP[a + d:end, :], X[i:i + 1, :])
                pos = new_pos

            new_index.append(pos)
            index = new_index

    return final(XP)
