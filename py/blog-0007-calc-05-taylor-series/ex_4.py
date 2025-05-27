#!/usr/bin/env python3
"""
exercise_4_adaptive_degree.py
-------------------------------------------------
Find the smallest Maclaurin degree n such that

    |e^x − T_n(x)| < 1e‑4,

where  T_n(x) = Σ_{k=0}^n x^k / k!.  Test for
x = 0.5, 1, 2, 5 and print the required n.
"""

from math import exp

TOL = 1e-4
X_TEST = [0.5, 1.0, 2.0, 5.0]


def min_degree_exp(x, tol=TOL, max_n=200):
    """Return the smallest n with |e^x - T_n(x)| < tol."""
    partial = 0.0
    term = 1.0  # x^0 / 0!
    for n in range(max_n + 1):
        partial += term
        if abs(exp(x) - partial) < tol:
            return n
        # next term: x^{n+1}/(n+1)!
        term *= x / (n + 1)
    raise RuntimeError("max_n too small for convergence")


print(f"Tolerance = {TOL}\n")
for x in X_TEST:
    n = min_degree_exp(x)
    print(f"x = {x:<3} →  smallest n = {n}")
