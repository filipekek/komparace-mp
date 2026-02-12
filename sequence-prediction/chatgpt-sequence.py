#!/usr/bin/env python3
"""
Sequence Predictor (pure Python; numpy optional but not required)

- Reads at least 3 numbers from terminal (comma and/or space separated).
- Tries to identify the sequence type (when meaningful).
- Outputs:
  * Type
  * Equation (closed form when available; otherwise a recurrence)
  * Next 5 numbers

Notes:
- For exact integer inputs, the code uses exact rational arithmetic (fractions) internally.
- For non-integer inputs, the code uses floats with a small tolerance.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import List, Tuple, Optional, Union
import time
Number = Union[Fraction, float]


def _is_close(a: Number, b: Number, tol: float) -> bool:
    if isinstance(a, Fraction) and isinstance(b, Fraction):
        return a == b
    return abs(float(a) - float(b)) <= tol


def _all_close(values: List[Number], tol: float) -> bool:
    if not values:
        return True
    first = values[0]
    return all(_is_close(v, first, tol) for v in values[1:])


def _format_number(x: Number) -> str:
    if isinstance(x, Fraction):
        if x.denominator == 1:
            return str(x.numerator)
        return f"{x.numerator}/{x.denominator}"
    # float formatting: keep readable, avoid trailing zeros
    if math.isfinite(x):
        s = f"{x:.12g}"
        return s
    return str(x)


def _format_sequence(seq: List[Number]) -> str:
    return " ".join(_format_number(x) for x in seq)


def parse_input_numbers(raw: str) -> List[Number]:
    # Accept commas and/or spaces as separators
    cleaned = raw.replace(",", " ").strip()
    parts = [p for p in cleaned.split() if p]

    if len(parts) < 3:
        raise ValueError("Please enter at least 3 numbers.")

    # Try to parse as integers first; if any fails, parse all as floats
    ints: List[int] = []
    all_int = True
    for p in parts:
        try:
            # Allow leading +/-, but reject things like "1.0" as int
            if any(ch in p for ch in ".eE"):
                all_int = False
                break
            ints.append(int(p))
        except ValueError:
            all_int = False
            break

    if all_int:
        return [Fraction(v, 1) for v in ints]

    # Fallback: float parsing for all tokens
    floats: List[float] = []
    for p in parts:
        try:
            floats.append(float(p))
        except ValueError as e:
            raise ValueError(f"Could not parse '{p}' as a number.") from e
    return floats


def is_constant(seq: List[Number], tol: float) -> bool:
    return _all_close(seq, tol)


def is_arithmetic(seq: List[Number], tol: float) -> Tuple[bool, Optional[Number]]:
    if len(seq) < 2:
        return False, None
    diffs = [seq[i + 1] - seq[i] for i in range(len(seq) - 1)]
    if _all_close(diffs, tol):
        return True, diffs[0]
    return False, None


def is_geometric(seq: List[Number], tol: float) -> Tuple[bool, Optional[Number]]:
    # Geometric requires consistent ratio where defined.
    # Handle zeros carefully: if a term is 0, next must be 0 for constant ratio (unless ratio undefined).
    if len(seq) < 2:
        return False, None

    ratios: List[Number] = []
    for i in range(len(seq) - 1):
        a, b = seq[i], seq[i + 1]
        if _is_close(a, 0, tol):
            # If a is (near) zero, then b must also be (near) zero to keep a consistent ratio for all later terms.
            if not _is_close(b, 0, tol):
                return False, None
            # ratio is not informative here; skip
            continue
        ratios.append(b / a)

    if not ratios:
        # All transitions are 0->0, treat as constant (handled earlier) not geometric
        return False, None

    if _all_close(ratios, tol):
        return True, ratios[0]
    return False, None


def try_order2_linear_recurrence(seq: List[Number], tol: float) -> Tuple[bool, Optional[Tuple[Number, Number]]]:
    """
    Try to fit a_n = p*a_{n-1} + q*a_{n-2} for all n>=3.

    We solve p,q using first two equations (n=3 and n=4) when possible:
      a3 = p*a2 + q*a1
      a4 = p*a3 + q*a2

    If that system is singular or too short, we attempt a simpler fit using one equation plus verification.
    """
    if len(seq) < 3:
        return False, None
    if len(seq) == 3:
        # Infinite solutions; not enough to identify. Require at least 4 to solve uniquely.
        return False, None

    a1, a2, a3, a4 = seq[0], seq[1], seq[2], seq[3]

    # Solve linear system:
    # [a2 a1] [p] = [a3]
    # [a3 a2] [q]   [a4]
    det = a2 * a2 - a1 * a3
    if _is_close(det, 0, tol):
        # Singular; attempt a fallback:
        # If a2 != 0, try q=0, p=a3/a2 and verify; else if a1 != 0 try p=0, q=a3/a1
        candidates = []
        if not _is_close(a2, 0, tol):
            p = a3 / a2
            q = 0 if isinstance(seq[0], float) else Fraction(0, 1)
            candidates.append((p, q))
        if not _is_close(a1, 0, tol):
            p = 0 if isinstance(seq[0], float) else Fraction(0, 1)
            q = a3 / a1
            candidates.append((p, q))

        for p, q in candidates:
            ok = True
            for i in range(2, len(seq)):
                pred = p * seq[i - 1] + q * seq[i - 2]
                if not _is_close(seq[i], pred, tol):
                    ok = False
                    break
            if ok:
                return True, (p, q)
        return False, None

    p = (a3 * a2 - a1 * a4) / det
    q = (a2 * a4 - a3 * a3) / det

    # Verify across sequence
    for i in range(2, len(seq)):
        pred = p * seq[i - 1] + q * seq[i - 2]
        if not _is_close(seq[i], pred, tol):
            return False, None

    return True, (p, q)


def build_difference_table(seq: List[Number]) -> List[List[Number]]:
    table = [seq[:]]
    while len(table[-1]) > 1:
        prev = table[-1]
        table.append([prev[i + 1] - prev[i] for i in range(len(prev) - 1)])
    return table


def detect_polynomial_by_differences(seq: List[Number], tol: float) -> Tuple[bool, Optional[int], Optional[List[List[Number]]]]:
    """
    If forward differences become constant at some level d, then the sequence is
    a polynomial of degree d.

    Returns: (ok, degree, difference_table)
    """
    table = build_difference_table(seq)
    # level 0: original, level 1: first diffs, ...
    for level in range(1, len(table)):
        if _all_close(table[level], tol):
            return True, level, table
    return False, None, None


def extend_by_differences(table: List[List[Number]], k: int) -> List[Number]:
    """
    Extend sequence by k terms using the forward-difference table.
    Works best for polynomial sequences (when some difference row is constant).
    """
    # Make a deep-ish copy of "last elements" per row
    last = [row[:] for row in table]

    for _ in range(k):
        # Append new value at bottom row (constant difference row) by repeating its last value
        # For general case, bottom row is length 1; we can keep it constant.
        last[-1].append(last[-1][-1])

        # Move upward: new last element = previous last element + new last element of row below
        for r in range(len(last) - 2, -1, -1):
            last[r].append(last[r][-1] + last[r + 1][-1])

    # Return only the newly generated terms from the top row
    original_len = len(table[0])
    return last[0][original_len:]


def binom(n: int, k: int) -> int:
    return math.comb(n, k)


def polynomial_newton_equation_from_differences(table: List[List[Number]]) -> str:
    """
    For n starting at 1, Newton forward form:
      a(n) = sum_{k=0..d} C(n-1, k) * Δ^k a(1)

    where Δ^k a(1) are the first entries in each difference row.
    """
    d = len(table) - 1
    terms = []
    for k in range(0, d + 1):
        coeff = table[k][0]
        if isinstance(coeff, Fraction) and coeff == 0:
            continue
        if isinstance(coeff, float) and abs(coeff) < 1e-15:
            continue

        c = _format_number(coeff)
        if k == 0:
            terms.append(f"{c}")
        else:
            terms.append(f"{c}*C(n-1,{k})")
    if not terms:
        return "a(n) = 0"
    return "a(n) = " + " + ".join(terms) + "    (where C is the binomial coefficient)"


def predict_next(seq: List[Number]) -> Tuple[str, str, List[Number]]:
    """
    Determine best-matching sequence type and predict next 5 terms.
    Returns (type_str, equation_str, next_terms)
    """
    # Tolerance: exact for Fractions, small for floats
    tol = 0.0 if isinstance(seq[0], Fraction) else 1e-9

    # 1) Constant
    if is_constant(seq, tol):
        c = seq[0]
        next_terms = [c] * 5
        return "Constant", f"a(n) = {_format_number(c)}", next_terms

    # 2) Arithmetic
    ok, d = is_arithmetic(seq, tol)
    if ok and d is not None:
        a1 = seq[0]
        # a(n) = a1 + (n-1)d
        equation = f"a(n) = {_format_number(a1)} + (n-1)*{_format_number(d)}"
        last = seq[-1]
        next_terms = [last + d * i for i in range(1, 6)]
        return "Arithmetic", equation, next_terms

    # 3) Geometric
    ok, r = is_geometric(seq, tol)
    if ok and r is not None:
        a1 = seq[0]
        equation = f"a(n) = {_format_number(a1)}*({_format_number(r)})^(n-1)"
        next_terms = []
        current = seq[-1]
        for _ in range(5):
            current = current * r
            next_terms.append(current)
        return "Geometric", equation, next_terms

    # 4) Order-2 linear recurrence (covers Fibonacci-like and more)
    ok, pq = try_order2_linear_recurrence(seq, tol)
    if ok and pq is not None:
        p, q = pq
        equation = (
            f"a(n) = {_format_number(p)}*a(n-1) + {_format_number(q)}*a(n-2),  "
            f"with a(1)={_format_number(seq[0])}, a(2)={_format_number(seq[1])}"
        )
        next_terms = []
        a_nm2, a_nm1 = seq[-2], seq[-1]
        for _ in range(5):
            a_n = p * a_nm1 + q * a_nm2
            next_terms.append(a_n)
            a_nm2, a_nm1 = a_nm1, a_n
        return "Linear Recurrence (order 2)", equation, next_terms

    # 5) Polynomial via finite differences (quadratic, cubic, ...)
    ok, degree, table = detect_polynomial_by_differences(seq, tol)
    if ok and degree is not None and table is not None:
        next_terms = extend_by_differences(table, 5)
        equation = polynomial_newton_equation_from_differences(table[: degree + 1])
        name = "Quadratic" if degree == 2 else ("Cubic" if degree == 3 else f"Polynomial (degree {degree})")
        return name, equation, next_terms

    # 6) Fallback: exact polynomial interpolation of degree (m-1) for integer inputs (guaranteed match)
    # For floats, this can be numerically unstable; we avoid it and instead provide an "unknown" response.
    if isinstance(seq[0], Fraction):
        # Lagrange interpolation to predict next values at n=m+1..m+5, with n starting at 1
        m = len(seq)

        def lagrange_value(x: int) -> Fraction:
            total = Fraction(0, 1)
            for i in range(1, m + 1):
                yi = seq[i - 1]
                num = Fraction(1, 1)
                den = Fraction(1, 1)
                for j in range(1, m + 1):
                    if j == i:
                        continue
                    num *= Fraction(x - j, 1)
                    den *= Fraction(i - j, 1)
                total += yi * (num / den)
            return total

        next_terms = [lagrange_value(m + t) for t in range(1, 6)]
        equation = (
            "a(n) is the unique polynomial of degree <= "
            f"{m-1} that fits the provided terms (via Lagrange interpolation)."
        )
        return f"Polynomial Interpolation (degree ≤ {m-1})", equation, next_terms

    # Unknown / unsupported pattern
    return (
        "Unknown / Not confidently classified",
        "Could not match common families (constant/arithmetic/geometric/order-2 recurrence/polynomial differences).",
        []
    )


def main() -> None:
    print("Input at least 3 numerals separated by comma or space:")
    raw = input("> ")
    t1 = time.perf_counter()
    try:
        seq = parse_input_numbers(raw)
    except ValueError as e:
        print(f"Error: {e}")
        return

    type_str, equation_str, next_terms = predict_next(seq)

    print(f"Type: {type_str}")
    print(f"Equation: {equation_str}")
    if next_terms:
        print("Next 5 numbers:", _format_sequence(next_terms))
    else:
        print("Next 5 numbers: (not available)")
    t2 = time.perf_counter()
    print(t2-t1)

    # Optional: show what was understood
    # print("Given sequence:", _format_sequence(seq))


if __name__ == "__main__":
    main()
