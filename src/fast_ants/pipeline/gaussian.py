"""Host Gaussian coefficients adapted from ITK 5.4.7, NumFOCUS, Apache-2.0."""
import math
import numpy as np
def recursive_gaussian_coefficients(sigma_vox):
    if not np.isfinite(sigma_vox) or sigma_vox <= 0:
        raise ValueError("Recursive Gaussian sigma must be finite and positive")
    a1, b1, a2, b2 = 1.3530, 1.8151, -.3531, .0902
    s1, s2 = math.sin(.6681/sigma_vox), math.sin(2.0787/sigma_vox)
    c1, c2 = math.cos(.6681/sigma_vox), math.cos(2.0787/sigma_vox)
    e1, e2 = math.exp(-1.3932/sigma_vox), math.exp(-1.3732/sigma_vox)
    denominator = np.array([-2*(e2*c2+e1*c1),
                            4*c2*c1*e1*e2 + e1*e1 + e2*e2,
                            -2*c1*e1*e2*e2 - 2*c2*e2*e1*e1,
                            e1*e1*e2*e2])
    numerator = np.array([a1+a2,
                          e2*(b2*s2-(a2+2*a1)*c2) + e1*(b1*s1-(a1+2*a2)*c1),
                          2*e1*e2*((a1+a2)*c2*c1-b1*c2*s1-b2*c1*s2)+a2*e1*e1+a1*e2*e2,
                          e2*e1*e1*(b2*s2-a2*c2)+e1*e2*e2*(b1*s1-a1*c1)])
    total_denominator = 1 + sum(denominator)
    normalization = 2*sum(numerator)/total_denominator - numerator[0]
    numerator /= normalization
    anticausal = np.r_[numerator[1:]-denominator[:3]*numerator[0], -denominator[3]*numerator[0]]
    return (numerator, denominator, anticausal,
            denominator*sum(numerator)/total_denominator,
            denominator*sum(anticausal)/total_denominator)


def _polynomial(x, coefficients):
    output = 0.0
    for coefficient in reversed(coefficients):
        output = coefficient + x*output
    return output


def _bessel0(value):
    value = abs(value)
    if value < 3.75:
        return _polynomial((value/3.75)**2, [1, 3.5156229, 3.0899424, 1.2067492, .2659732, .0360768, .0045813])
    return math.exp(value)/math.sqrt(value)*_polynomial(3.75/value,
        [.39894228, .01328592, .00225319, -.00157565, .00916281, -.02057706, .02635537, -.01647633, .00392377])


def _bessel1(value):
    value = abs(value)
    if value < 3.75:
        return value*_polynomial((value/3.75)**2, [.5, .87890594, .51498869, .15084934, .02658733, .00301532, .00032411])
    return math.exp(value)/math.sqrt(value)*_polynomial(3.75/value,
        [.39894228, -.03988024, -.00362018, .00163801, -.01031555, .02282967, -.02895312, .01787654, -.00420059])


def discrete_gaussian_kernel(variance, maximum_width=32, maximum_error=.001):
    if not np.isfinite(variance) or variance < 0 or variance > 500:
        raise ValueError("Discrete Gaussian variance must be finite in [0,500] for this reference implementation")
    if not 0 < maximum_error < 1 or maximum_width < 1:
        raise ValueError("Invalid Gaussian kernel truncation settings")
    if variance == 0:
        return np.array([1.0])
    scale = math.exp(-variance)
    half = [scale*_bessel0(variance), scale*_bessel1(variance)]
    total = half[0] + 2*half[1]
    order = 2
    while total < 1-maximum_error:
        coefficient = scale*_bessel_n(order, variance)
        half.append(coefficient)
        total += 2*coefficient
        if coefficient <= 0 or len(half) > maximum_width:
            break
        order += 1
    half = np.array(half)/total
    return np.r_[half[:0:-1], half]



def _bessel_n(n, value):
    if value == 0:
        return 0.0
    qi, qip, accumulator = 1.0, 0.0, 0.0
    for j in range(2*(n+int(math.sqrt(40*n))), 0, -1):
        qim = qip + j*(2/abs(value))*qi
        qip, qi = qi, qim
        if abs(qi) > 1e10:
            accumulator *= 1e-10
            qi *= 1e-10
            qip *= 1e-10
        if j == n:
            accumulator = qip
    return accumulator*_bessel0(value)/qi

