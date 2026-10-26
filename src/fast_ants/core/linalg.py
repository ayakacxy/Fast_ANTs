"""Small FP64 geometry inverses using LINPACK's bidiagonal SVD ordering.

ITK's Matrix::GetInverse uses VNL/LINPACK SVD, not a general LU inverse.
For Mattes' discontinuous fixed bins, even the different rounding of those
two valid inverses can change the registration trajectory. This independent
3x3 specialization retains the scaled norms and QR control of LINPACK DSVDC.
It performs ordinary hardware FP64 arithmetic; no upstream library is called.
"""

import math

import numpy as np
from numba import njit

from fast_ants.core.arithmetic64 import fma64


@njit(cache=True, fastmath={"contract"})
def _norm(values, first):
    if 3-first == 1:
        return abs(values[first])
    scale, ssq = 0.0, 1.0
    for i in range(first, 3):
        value = abs(values[i])
        if value != 0.0:
            if scale < value:
                ratio = scale/value
                ssq = ssq*(ratio*ratio)+1.0
                scale = value
            else:
                ratio = value/scale
                ssq += ratio*ratio
    return scale*math.sqrt(ssq)


@njit(cache=True, fastmath={"contract"})
def _dot(a, b, first):
    total = 0.0
    for i in range(first, 3):
        total += a[i]*b[i]
    return total


@njit(cache=True, fastmath={"contract"})
def _givens(a, b):
    scale = abs(a)+abs(b)
    if scale == 0.0:
        return 0.0, 1.0, 0.0
    roe = a if abs(a) > abs(b) else b
    aa, bb = a/scale, b/scale
    r = scale*math.sqrt(aa*aa+bb*bb)
    r = (1.0 if roe >= 0.0 else -1.0)*r
    return r, a/r, b/r


@njit(cache=True, fastmath={"contract"})
def _rotate(matrix, left, right, cs, sn):
    for i in range(3):
        a, b = matrix[i, left], matrix[i, right]
        temporary = cs*a+sn*b
        matrix[i, right] = cs*b-sn*a
        matrix[i, left] = temporary


@njit(cache=True, fastmath={"contract"})
def svd3(matrix):
    """3x3 Householder reduction followed by implicit shifted QR."""
    x = matrix.copy()
    u, v = np.zeros((3, 3)), np.zeros((3, 3))
    s, e, work = np.zeros(3), np.zeros(3), np.zeros(3)
    # Two column reflectors, and one row reflector.
    for l in range(2):
        s[l] = _norm(x[:, l], l)
        if s[l] != 0.0:
            if x[l, l] != 0.0:
                s[l] = (1.0 if x[l, l] >= 0.0 else -1.0)*s[l]
            factor = 1.0/s[l]
            for i in range(l, 3):
                x[i, l] *= factor
            x[l, l] += 1.0
        s[l] = -s[l]
        for j in range(l+1, 3):
            if s[l] != 0.0:
                t = -_dot(x[:, l], x[:, j], l)/x[l, l]
                for i in range(l, 3):
                    x[i, j] += t*x[i, l]
            e[j] = x[l, j]
        for i in range(l, 3):
            u[i, l] = x[i, l]
        if l == 0:
            e[l] = _norm(e, l+1)
            if e[l] != 0.0:
                if e[l+1] != 0.0:
                    e[l] = (1.0 if e[l+1] >= 0.0 else -1.0)*e[l]
                factor = 1.0/e[l]
                for i in range(l+1, 3):
                    e[i] *= factor
                e[l+1] += 1.0
            e[l] = -e[l]
            if e[l] != 0.0:
                for i in range(l+1, 3):
                    work[i] = 0.0
                for j in range(l+1, 3):
                    for i in range(l+1, 3):
                        work[i] += e[j]*x[i, j]
                for j in range(l+1, 3):
                    t = -e[j]/e[l+1]
                    for i in range(l+1, 3):
                        x[i, j] += t*work[i]
            for i in range(l+1, 3):
                v[i, l] = e[i]
    s[2], e[1], e[2] = x[2, 2], x[1, 2], 0.0
    u[2, 2] = 1.0
    for l in range(1, -1, -1):
        if s[l] != 0.0:
            for j in range(l+1, 3):
                t = -_dot(u[:, l], u[:, j], l)/u[l, l]
                for i in range(l, 3):
                    u[i, j] += t*u[i, l]
            for i in range(l, 3):
                u[i, l] *= -1.0
            u[l, l] += 1.0
            for i in range(l):
                u[i, l] = 0.0
        else:
            for i in range(3):
                u[i, l] = 0.0
            u[l, l] = 1.0
    for l in range(2, -1, -1):
        if l == 0 and e[l] != 0.0:
            for j in range(l+1, 3):
                t = -_dot(v[:, l], v[:, j], l+1)/v[l+1, l]
                for i in range(l+1, 3):
                    v[i, j] += t*v[i, l]
        for i in range(3):
            v[i, l] = 0.0
        v[l, l] = 1.0

    # Keep 1-based scalar indices for the small QR state machine.
    m, iteration = 3, 0
    while m != 0:
        if iteration >= 1000:
            raise ValueError("Geometry SVD did not converge")
        l = m-1
        while l != 0:
            test = abs(s[l-1])+abs(s[l])
            if test+abs(e[l-1]) == test:
                e[l-1] = 0.0
                break
            l -= 1
        if l == m-1:
            case = 4
        else:
            ls = m
            while ls != l:
                test = 0.0
                if ls != m:
                    test += abs(e[ls-1])
                if ls != l+1:
                    test += abs(e[ls-2])
                if test+abs(s[ls-1]) == test:
                    s[ls-1] = 0.0
                    break
                ls -= 1
            if ls == l:
                case = 3
            elif ls == m:
                case = 1
            else:
                case, l = 2, ls
        l += 1
        if case == 1:
            f = e[m-2]
            e[m-2] = 0.0
            for k in range(m-1, l-1, -1):
                s[k-1], cs, sn = _givens(s[k-1], f)
                if k != l:
                    f = -sn*e[k-2]
                    e[k-2] *= cs
                _rotate(v, k-1, m-1, cs, sn)
        elif case == 2:
            f = e[l-2]
            e[l-2] = 0.0
            for k in range(l, m+1):
                s[k-1], cs, sn = _givens(s[k-1], f)
                f = -sn*e[k-1]
                e[k-1] *= cs
                _rotate(u, k-1, l-2, cs, sn)
        elif case == 3:
            scale = max(abs(s[m-1]), abs(s[m-2]), abs(e[m-2]), abs(s[l-1]), abs(e[l-1]))
            sm, smm1 = s[m-1]/scale, s[m-2]/scale
            emm1, sl, el = e[m-2]/scale, s[l-1]/scale, e[l-1]/scale
            b = fma64(smm1+sm, smm1-sm, emm1*emm1)/2.0
            c = (sm*emm1)*(sm*emm1)
            shift = 0.0
            if b != 0.0 or c != 0.0:
                shift = math.sqrt(fma64(b, b, c))
                if b < 0.0:
                    shift = -shift
                shift = c/(b+shift)
            f, g = fma64(sl+sm, sl-sm, shift), sl*el
            for k in range(l, m):
                f, cs, sn = _givens(f, g)
                if k != l:
                    e[k-2] = f
                f = fma64(cs, s[k-1], sn*e[k-1])
                e[k-1] = fma64(cs, e[k-1], -sn*s[k-1])
                g = sn*s[k]
                s[k] *= cs
                _rotate(v, k-1, k, cs, sn)
                f, cs, sn = _givens(f, g)
                s[k-1] = f
                f = fma64(cs, e[k-1], sn*s[k])
                s[k] = fma64(-sn, e[k-1], cs*s[k])
                g = sn*e[k]
                e[k] *= cs
                _rotate(u, k-1, k, cs, sn)
            e[m-2] = f
            iteration += 1
        else:
            if s[l-1] < 0.0:
                s[l-1] = -s[l-1]
                for i in range(3):
                    v[i, l-1] *= -1.0
            while l != 3 and s[l-1] < s[l]:
                s[l-1], s[l] = s[l], s[l-1]
                for i in range(3):
                    v[i, l-1], v[i, l] = v[i, l], v[i, l-1]
                    u[i, l-1], u[i, l] = u[i, l], u[i, l-1]
                l += 1
            iteration, m = 0, m-1
    return u, s, v


@njit(cache=True, fastmath={"contract"})
def inverse3(matrix):
    """V * (1/S) * U.T, with bounded shape and no large-array SVD call."""
    u, singular, v = svd3(matrix)
    if singular[2] <= 0.0:
        raise ValueError("Geometry matrix is singular")
    diagonal = np.zeros((3, 3))
    for i in range(3):
        diagonal[i, i] = 1.0/singular[i]
    scaled_v, result = np.zeros((3, 3)), np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                scaled_v[i, j] += v[i, k]*diagonal[k, j]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                result[i, j] += scaled_v[i, k]*u[j, k]
    return result
