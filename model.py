"""
A/B Testing & Causal Inference Toolkit

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - standard_normal_cdf
import math
import numpy as np

def standard_normal_cdf(z):
    # TODO: return P(Z <= z) for a standard normal Z, supporting float or numpy array input
    z = z / np.sqrt(2.0)
    z = np.vectorize(math.erf)(z)
    z = 0.5*(1+z)
    return z

# Step 2 - standard_normal_ppf
import math

def standard_normal_ppf(p):
    """
    Return z such that Phi(z) = p for p in (0, 1) using Acklam's 
    rational approximation for the inverse standard normal CDF.
    """
    # Boundary / Edge-case checks
    if p <= 0 or p >= 1:
        raise ValueError("p must be strictly between 0 and 1")

    p_low = 0.02425
    p_high = 1 - p_low

    # Coefficients for the Central Region (0.02425 <= p <= 0.97575)
    COEFF_A = [
        -3.969683028665376e+01,
         2.209460984245205e+02,
        -2.759285104469687e+02,
         1.383577518672690e+02,
        -3.066479806614716e+01,
         2.506628277459239e+00
    ]
    COEFF_B = [
        -5.447609879822406e+01,
         1.615858368580409e+02,
        -1.556989798598866e+02,
         6.680131188771972e+01,
        -1.328068155288572e+01
    ]

    # Coefficients for the Tail Regions (p < 0.02425 or p > 0.97575)
    COEFF_C = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
         4.374664141464968e+00,
         2.938163982698783e+00
    ]
    COEFF_D = [
         7.784695709041462e-03,
         3.224671290700398e-01,
         2.445134137142996e+00,
         3.754408661907416e+00
    ]

    # Lower Tail Region
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        num = COEFF_C[0]
        for coeff in COEFF_C[1:]:
            num = num * q + coeff
        denom = COEFF_D[0]
        for coeff in COEFF_D[1:]:
            denom = denom * q + coeff
        denom = denom * q + 1.0  # Implicit trailing +1 in Acklam's tail denominator
        return num / denom

    # Upper Tail Region
    elif p > p_high:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        num = COEFF_C[0]
        for coeff in COEFF_C[1:]:
            num = num * q + coeff
        denom = COEFF_D[0]
        for coeff in COEFF_D[1:]:
            denom = denom * q + coeff
        denom = denom * q + 1.0  # Implicit trailing +1 in Acklam's tail denominator
        return -num / denom

    # Central Region
    else:
        q = p - 0.5
        r = q * q
        num = COEFF_A[0]
        for coeff in COEFF_A[1:]:
            num = num * r + coeff
        denom = COEFF_B[0]
        for coeff in COEFF_B[1:]:
            denom = denom * r + coeff
        denom = denom * r + 1.0  # Implicit trailing +1 in Acklam's central denominator
        return q * num / denom

