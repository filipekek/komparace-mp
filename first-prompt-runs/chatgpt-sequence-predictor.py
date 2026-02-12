from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isclose
from typing import List, Optional, Sequence, Tuple, Union


Number = Union[Fraction, float]


@dataclass(frozen=True)
class SequenceModel:
    kind: str                   # e.g., "Arithmetic", "Geometric", "Polynomial (degree 2)", ...
    equation: str               # human-readable closed form / recurrence
    next_terms: List[Number]    # predicted next k terms
    notes: str = ""             # optional details / caveats


def _is_fractional(x: Number) -> bool:
    return isinstance(x, Fraction)


def _to_fraction_if_reasonable(tokens: Sequence[str]) -> List[Number]:
    """
    Parse user tokens into numbers.
    - If all tokens parse cleanly as integers/finite decimals, use Fraction for exactness.
    - Otherwise, fall back to float.
    """
    fracs: List[Fraction] = []
    try:
        for t in tokens:
            t = t.strip()
            if not t:
                raise ValueError("empty token")
            fracs.append(Fraction(t))  # Fraction can parse "3", "-2", "1.25", "5/7"
        return fracs
    except Exception:
        # Fallback: float parsing
        vals: List[float] = []
        for t in tokens:
            t = t.strip()
            if not t:
                raise ValueError("Empty token in input.")
            vals.append(float(t))
        return vals


def _almost_equal(a: Number, b: Number, *, tol: float = 1e-9) -> bool:
    if _is_fractional(a) and _is_fractional(b):
        return a == b
    # Compare as floats with tolerance
    return isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def _all_almost_equal(vals: Sequence[Number], *, tol: float = 1e-9) -> bool:
    if not vals:
        return True
    first = vals[0]
    return all(_almost_equal(v, first, tol=tol) for v in vals[1:])


def _safe_div(a: Number, b: Number) -> Optional[Number]:
    if _almost_equal(b, 0):
        return None
    if _is_fractional(a) and _is_fractional(b):
        return a / b
    return float(a) / float(b)


def _predict_arithmetic(seq: Sequence[Number], k: int) -> Optional[SequenceModel]:
    if len(seq) < 2:
        return None
    diffs = [seq[i + 1] - seq[i] for i in range(len(seq) - 1)]
    if not _all_almost_equal(diffs):
        return None

    d = diffs[0]
    a0 = seq[0]
    n0 = 0  # indexing from n=0
    # next terms
    last = seq[-1]
    next_terms = [last + (i + 1) * d for i in range(k)]

    eq = f"a(n) = {a0} + {d}·n   (n≥0, a(0)={a0})"
    if _almost_equal(d, 0):
        kind = "Constant (Arithmetic with d=0)"
    else:
        kind = "Arithmetic"

    return SequenceModel(kind=kind, equation=eq, next_terms=next_terms)


def _predict_geometric(seq: Sequence[Number], k: int) -> Optional[SequenceModel]:
    if len(seq) < 2:
        return None

    # Special case: all zeros => geometric with r arbitrary; choose r=0
    if all(_almost_equal(x, 0) for x in seq):
        next_terms = [seq[-1] for _ in range(k)]
        return SequenceModel(
            kind="Geometric (all zeros)",
            equation="a(n) = 0",
            next_terms=next_terms,
        )

    ratios: List[Number] = []
    for i in range(len(seq) - 1):
        r = _safe_div(seq[i + 1], seq[i])
        if r is None:
            # If seq[i] is 0, geometric ratio is undefined unless next is also 0.
            if _almost_equal(seq[i], 0) and _almost_equal(seq[i + 1], 0):
                # This pair doesn't constrain r; skip it.
                continue
            return None
        ratios.append(r)

    if not ratios:
        # e.g., sequence like [0,0,0,...] handled above; other patterns cannot define a unique ratio.
        return None

    if not _all_almost_equal(ratios):
        return None

    r = ratios[0]
    a0 = seq[0]
    last = seq[-1]
    next_terms = [last * (r ** (i + 1)) for i in range(k)] if _is_fractional(r) else [float(last) * (float(r) ** (i + 1)) for i in range(k)]
    eq = f"a(n) = {a0}·({r})^n   (n≥0, a(0)={a0})"
    kind = "Geometric"
    return SequenceModel(kind=kind, equation=eq, next_terms=next_terms)


def _finite_differences(seq: Sequence[Number]) -> List[List[Number]]:
    """
    Return table of forward differences:
    levels[0] = original
    levels[1] = first differences
    ...
    """
    levels: List[List[Number]] = [list(seq)]
    while len(levels[-1]) >= 2:
        prev = levels[-1]
        nxt = [prev[i + 1] - prev[i] for i in range(len(prev) - 1)]
        levels.append(nxt)
    return levels


def _infer_polynomial_degree(seq: Sequence[Number], max_degree: int = 6) -> Optional[int]:
    """
    If forward differences become constant at some level d, infer degree d.
    Uses exact equality for Fractions; tolerance for floats.
    """
    levels = _finite_differences(seq)
    # levels[0] length n; levels[d] length n-d
    for d in range(1, min(max_degree, len(seq) - 1) + 1):
        if _all_almost_equal(levels[d]):
            return d
    return None


def _gaussian_solve(A: List[List[Fraction]], b: List[Fraction]) -> Optional[List[Fraction]]:
    """
    Solve A x = b for small systems using Gaussian elimination over Fractions.
    Returns None if singular.
    """
    n = len(A)
    # augment
    M = [row[:] + [b[i]] for i, row in enumerate(A)]

    # forward elimination
    for col in range(n):
        pivot = None
        for r in range(col, n):
            if M[r][col] != 0:
                pivot = r
                break
        if pivot is None:
            return None
        if pivot != col:
            M[col], M[pivot] = M[pivot], M[col]

        # normalize pivot row
        pv = M[col][col]
        for c in range(col, n + 1):
            M[col][c] /= pv

        # eliminate below
        for r in range(col + 1, n):
            factor = M[r][col]
            if factor == 0:
                continue
            for c in range(col, n + 1):
                M[r][c] -= factor * M[col][c]

    # back substitution
    x = [Fraction(0) for _ in range(n)]
    for r in range(n - 1, -1, -1):
        s = M[r][n]
        for c in range(r + 1, n):
            s -= M[r][c] * x[c]
        x[r] = s  # M[r][r] is 1
    return x


def _poly_fit_coeffs_fraction(seq: Sequence[Fraction], degree: int) -> Optional[List[Fraction]]:
    """
    Fit polynomial p(n) = c0 + c1*n + ... + cD*n^D through points (n, a_n) for n=0..degree.
    Uses a Vandermonde system on the first degree+1 points.
    """
    if len(seq) < degree + 1:
        return None
    m = degree + 1
    A: List[List[Fraction]] = []
    b: List[Fraction] = []
    for n in range(m):
        row = [Fraction(1)]
        for p in range(1, m):
            row.append(row[-1] * Fraction(n))
        A.append(row)
        b.append(seq[n])
    return _gaussian_solve(A, b)


def _poly_eval(coeffs: Sequence[Number], n: int) -> Number:
    # Horner's method
    if not coeffs:
        return Fraction(0)
    if all(_is_fractional(c) for c in coeffs):
        acc: Fraction = Fraction(0)
        for c in reversed(coeffs):
            acc = acc * n + c  # type: ignore[arg-type]
        return acc
    accf = 0.0
    for c in reversed(coeffs):
        accf = accf * float(n) + float(c)
    return accf


def _format_poly_equation(coeffs: Sequence[Number]) -> str:
    terms = []
    for p, c in enumerate(coeffs):
        if _almost_equal(c, 0):
            continue
        if p == 0:
            terms.append(f"{c}")
        elif p == 1:
            terms.append(f"{c}·n")
        else:
            terms.append(f"{c}·n^{p}")
    if not terms:
        return "a(n) = 0"
    return "a(n) = " + " + ".join(terms) + "   (n≥0, n is the index)"


def _predict_polynomial(seq: Sequence[Number], k: int) -> Optional[SequenceModel]:
    deg = _infer_polynomial_degree(seq)
    if deg is None:
        return None

    # If float input, we can still extrapolate using difference table directly (stable for small k),
    # but for the equation we prefer exact coefficients when possible.
    levels = _finite_differences(seq)

    # Extrapolate k steps using forward differences:
    # Repeatedly extend the last element of each difference level.
    # For a polynomial, the last difference at level deg is constant.
    levels_ext = [lvl[:] for lvl in levels]
    # Ensure we only rely on inferred degree being constant, not necessarily the very last level.
    for _ in range(k):
        # Extend from bottom (degree level) up to original
        for d in range(deg, 0, -1):
            last = levels_ext[d][-1]
            levels_ext[d].append(last)  # constant at degree d
        # Now rebuild upper levels: a new value at level d-1 is last(prev) + last(diff)
        for d in range(deg - 1, -1, -1):
            new_val = levels_ext[d][-1] + levels_ext[d + 1][-1]
            levels_ext[d].append(new_val)

    next_terms = levels_ext[0][-k:]

    # Try to compute a closed-form polynomial equation.
    equation = f"Polynomial of degree {deg} (closed form not computed)"
    notes = ""
    if all(_is_fractional(x) for x in seq):
        coeffs = _poly_fit_coeffs_fraction(seq, deg)  # type: ignore[arg-type]
        if coeffs is not None:
            equation = _format_poly_equation(coeffs)
        else:
            notes = "Polynomial degree inferred from finite differences, but coefficient solving was singular."
    else:
        # For floats, produce an equation via numeric fit on first deg+1 points (as floats).
        # This is mainly for readability; extrapolation above already uses differences.
        try:
            # Build float Vandermonde and solve via basic Gaussian elimination in float
            m = deg + 1
            A = [[(float(n) ** p) for p in range(m)] for n in range(m)]
            b = [float(seq[n]) for n in range(m)]
            coeffs = _gaussian_solve(
                [[Fraction(aij).limit_denominator(10**9) for aij in row] for row in A],
                [Fraction(bi).limit_denominator(10**9) for bi in b],
            )
            if coeffs is not None:
                equation = _format_poly_equation(coeffs)
                notes = "Coefficients shown as rational approximations of float input."
        except Exception:
            pass

    return SequenceModel(
        kind=f"Polynomial (degree {deg})",
        equation=equation,
        next_terms=next_terms,
        notes=notes,
    )


def _solve_2x2_fraction(
    a11: Fraction, a12: Fraction, a21: Fraction, a22: Fraction, b1: Fraction, b2: Fraction
) -> Optional[Tuple[Fraction, Fraction]]:
    det = a11 * a22 - a12 * a21
    if det == 0:
        return None
    x = (b1 * a22 - a12 * b2) / det
    y = (a11 * b2 - b1 * a21) / det
    return x, y


def _predict_linear_recurrence_order2(seq: Sequence[Number], k: int) -> Optional[SequenceModel]:
    """
    Try a(n) = p*a(n-1) + q*a(n-2), constant p,q.
    Needs at least 4 terms to uniquely solve in general.
    """
    if len(seq) < 4:
        return None

    # Prefer exact arithmetic with Fractions when possible.
    if all(_is_fractional(x) for x in seq):
        s = list(seq)  # type: ignore[assignment]
        a0, a1, a2, a3 = s[0], s[1], s[2], s[3]
        sol = _solve_2x2_fraction(
            a11=a1, a12=a0,
            a21=a2, a22=a1,
            b1=a2, b2=a3,
        )
        if sol is None:
            return None
        p, q = sol

        # Verify across the provided sequence.
        for i in range(2, len(s)):
            predicted = p * s[i - 1] + q * s[i - 2]
            if predicted != s[i]:
                return None

        # Predict next terms
        terms = s[:]
        for _ in range(k):
            terms.append(p * terms[-1] + q * terms[-2])
        next_terms = terms[-k:]

        eq = f"a(n) = {p}·a(n-1) + {q}·a(n-2),  with a(0)={a0}, a(1)={a1}"
        return SequenceModel(kind="Linear recurrence (order 2)", equation=eq, next_terms=next_terms)

    # Float path with tolerance
    s = [float(x) for x in seq]
    a0, a1, a2, a3 = s[0], s[1], s[2], s[3]
    det = a1 * a1 - a0 * a2
    if isclose(det, 0.0, abs_tol=1e-12, rel_tol=1e-12):
        return None
    p = (a2 * a1 - a0 * a3) / det
    q = (a1 * a3 - a2 * a2) / det

    for i in range(2, len(s)):
        predicted = p * s[i - 1] + q * s[i - 2]
        if not isclose(predicted, s[i], rel_tol=1e-9, abs_tol=1e-9):
            return None

    terms = s[:]
    for _ in range(k):
        terms.append(p * terms[-1] + q * terms[-2])
    next_terms = terms[-k:]
    eq = f"a(n) = {p}·a(n-1) + {q}·a(n-2),  with a(0)={a0}, a(1)={a1}"
    return SequenceModel(kind="Linear recurrence (order 2)", equation=eq, next_terms=next_terms, notes="Coefficients are approximate (float input).")


def predict_sequence(seq: Sequence[Number], *, next_k: int = 5) -> SequenceModel:
    """
    Detect a common sequence type and predict the next next_k terms.

    Tries, in order (simplest-first):
      1) Arithmetic
      2) Geometric
      3) Polynomial (via finite differences, degree up to 6)
      4) Linear recurrence of order 2

    If none match, returns a fallback model using a low-degree polynomial fit if possible,
    otherwise a simple "unknown" response.
    """
    if len(seq) < 3:
        raise ValueError("Please provide at least three numbers.")

    # 1) Arithmetic
    m = _predict_arithmetic(seq, next_k)
    if m:
        return m

    # 2) Geometric
    m = _predict_geometric(seq, next_k)
    if m:
        return m

    # 3) Polynomial (finite differences)
    m = _predict_polynomial(seq, next_k)
    if m:
        return m

    # 4) Order-2 linear recurrence
    m = _predict_linear_recurrence_order2(seq, next_k)
    if m:
        return m

    # Fallback: attempt polynomial interpolation with degree min(n-1, 3) for a reasonable guess
    deg = min(len(seq) - 1, 3)
    notes = "No standard type confidently matched. Using a low-degree polynomial interpolation as a heuristic."
    if all(_is_fractional(x) for x in seq):
        coeffs = _poly_fit_coeffs_fraction(seq, deg)  # type: ignore[arg-type]
        if coeffs is not None:
            next_terms = [_poly_eval(coeffs, len(seq) + i) for i in range(next_k)]
            eq = _format_poly_equation(coeffs)
            return SequenceModel(kind=f"Heuristic polynomial (degree {deg})", equation=eq, next_terms=next_terms, notes=notes)

    # Last-resort: constant continuation
    last = seq[-1]
    return SequenceModel(
        kind="Unknown (fallback)",
        equation="No reliable closed form detected; continuing the last value as a fallback.",
        next_terms=[last for _ in range(next_k)],
        notes=notes,
    )


def _parse_input_line(line: str) -> List[str]:
    # Accept space- or comma-separated, including mixed.
    raw = line.replace(",", " ").split()
    return [t.strip() for t in raw if t.strip()]


def main() -> None:
    print("Enter at least 3 numbers (space- or comma-separated). Examples: 1 4 7   or   1, 2, 4, 8")
    line = input("> ").strip()
    tokens = _parse_input_line(line)
    if len(tokens) < 3:
        raise SystemExit("Need at least 3 numbers.")

    seq = _to_fraction_if_reasonable(tokens)
    model = predict_sequence(seq, next_k=5)

    print("\nDetected type:", model.kind)
    print("Equation:", model.equation)
    print("Next 5 terms:", ", ".join(str(x) for x in model.next_terms))
    if model.notes:
        print("Notes:", model.notes)


if __name__ == "__main__":
    main()
