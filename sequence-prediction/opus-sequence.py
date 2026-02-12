#!/usr/bin/env python3
"""
Mathematical Sequence Predictor
Analyzes input sequences to determine their type, equation, and predict future terms.
"""

import math
from fractions import Fraction
from typing import List, Tuple, Optional, Callable
import time

def parse_input(user_input: str) -> List[float]:
    """Parse user input into a list of numbers."""
    # Replace commas with spaces and split
    cleaned = user_input.replace(',', ' ')
    parts = cleaned.split()
    
    numbers = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            # Try integer first, then float
            if '.' in part or 'e' in part.lower():
                numbers.append(float(part))
            else:
                numbers.append(int(part))
        except ValueError:
            raise ValueError(f"Invalid number: '{part}'")
    
    return numbers


def is_close(a: float, b: float, rel_tol: float = 1e-9, abs_tol: float = 1e-9) -> bool:
    """Check if two numbers are approximately equal."""
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def all_close(values: List[float], target: float) -> bool:
    """Check if all values in a list are close to a target."""
    return all(is_close(v, target) for v in values)


def differences(seq: List[float]) -> List[float]:
    """Calculate first differences of a sequence."""
    return [seq[i+1] - seq[i] for i in range(len(seq) - 1)]


def ratios(seq: List[float]) -> Optional[List[float]]:
    """Calculate ratios between consecutive terms. Returns None if division by zero."""
    result = []
    for i in range(len(seq) - 1):
        if seq[i] == 0:
            return None
        result.append(seq[i+1] / seq[i])
    return result


def format_number(n: float) -> str:
    """Format a number for display (integer if whole, otherwise float)."""
    if isinstance(n, int):
        return str(n)
    if is_close(n, round(n)):
        return str(int(round(n)))
    # Try to represent as fraction if reasonable
    try:
        frac = Fraction(n).limit_denominator(1000)
        if is_close(float(frac), n) and frac.denominator <= 100:
            if frac.denominator == 1:
                return str(frac.numerator)
            return f"{frac.numerator}/{frac.denominator}"
    except (ValueError, OverflowError):
        pass
    return f"{n:.6g}"


def format_coefficient(coef: float, is_first: bool = False, variable: str = "") -> str:
    """Format a coefficient for equation display."""
    if is_close(coef, 0):
        return ""
    
    sign = ""
    if coef < 0:
        sign = "-" if is_first else " - "
        coef = abs(coef)
    elif not is_first:
        sign = " + "
    
    if variable:
        if is_close(coef, 1):
            coef_str = ""
        else:
            coef_str = format_number(coef)
        return f"{sign}{coef_str}{variable}"
    else:
        return f"{sign}{format_number(coef)}"


# =============================================================================
# Sequence Detection Functions
# =============================================================================

def detect_constant(seq: List[float]) -> Optional[Tuple[str, str, Callable[[int], float]]]:
    """Detect constant sequence: a, a, a, ..."""
    if len(seq) < 2:
        return None
    if all_close(seq, seq[0]):
        c = seq[0]
        return (
            "Constant",
            f"a(n) = {format_number(c)}",
            lambda n, c=c: c
        )
    return None


def detect_arithmetic(seq: List[float]) -> Optional[Tuple[str, str, Callable[[int], float]]]:
    """Detect arithmetic sequence: a(n) = a + (n-1)d or a(n) = dn + c."""
    if len(seq) < 2:
        return None
    
    diffs = differences(seq)
    if not diffs:
        return None
    
    d = diffs[0]
    if all_close(diffs, d):
        # a(n) = d*n + c where a(1) = d + c, so c = a(1) - d
        a1 = seq[0]
        c = a1 - d
        
        # Build equation string
        if is_close(d, 0):
            eq = f"a(n) = {format_number(c)}"
        elif is_close(c, 0):
            eq = f"a(n) = {format_number(d)}n"
        else:
            eq = f"a(n) = {format_number(d)}n"
            if c > 0:
                eq += f" + {format_number(c)}"
            else:
                eq += f" - {format_number(abs(c))}"
        
        return (
            "Arithmetic",
            eq,
            lambda n, d=d, c=c: d * n + c
        )
    return None


def detect_geometric(seq: List[float]) -> Optional[Tuple[str, str, Callable[[int], float]]]:
    """Detect geometric sequence: a(n) = a * r^(n-1)."""
    if len(seq) < 2:
        return None
    
    # Check for zeros
    if any(is_close(x, 0) for x in seq[:-1]):
        return None
    
    r_list = ratios(seq)
    if r_list is None:
        return None
    
    r = r_list[0]
    if all_close(r_list, r):
        a = seq[0]
        eq = f"a(n) = {format_number(a)} × {format_number(r)}^(n-1)"
        
        return (
            "Geometric",
            eq,
            lambda n, a=a, r=r: a * (r ** (n - 1))
        )
    return None


def detect_quadratic(seq: List[float]) -> Optional[Tuple[str, str, Callable[[int], float]]]:
    """Detect quadratic sequence: a(n) = an² + bn + c."""
    if len(seq) < 3:
        return None
    
    # First differences
    d1 = differences(seq)
    # Second differences
    d2 = differences(d1)
    
    if not d2:
        return None
    
    # Check if second differences are constant
    if all_close(d2, d2[0]):
        # Second difference = 2a
        a = d2[0] / 2
        
        # First difference at n=1: d1[0] = a(2) - a(1) = 3a + b
        # So b = d1[0] - 3a
        b = d1[0] - 3 * a
        
        # a(1) = a + b + c, so c = seq[0] - a - b
        c = seq[0] - a - b
        
        # Build equation
        parts = []
        if not is_close(a, 0):
            if is_close(a, 1):
                parts.append("n²")
            elif is_close(a, -1):
                parts.append("-n²")
            else:
                parts.append(f"{format_number(a)}n²")
        
        if not is_close(b, 0):
            if is_close(b, 1):
                if parts:
                    parts.append(" + n")
                else:
                    parts.append("n")
            elif is_close(b, -1):
                parts.append(" - n")
            else:
                if parts:
                    if b > 0:
                        parts.append(f" + {format_number(b)}n")
                    else:
                        parts.append(f" - {format_number(abs(b))}n")
                else:
                    parts.append(f"{format_number(b)}n")
        
        if not is_close(c, 0) or not parts:
            if parts:
                if c > 0:
                    parts.append(f" + {format_number(c)}")
                else:
                    parts.append(f" - {format_number(abs(c))}")
            else:
                parts.append(format_number(c))
        
        eq = f"a(n) = {''.join(parts)}"
        
        return (
            "Quadratic",
            eq,
            lambda n, a=a, b=b, c=c: a * n * n + b * n + c
        )
    return None


def detect_cubic(seq: List[float]) -> Optional[Tuple[str, str, Callable[[int], float]]]:
    """Detect cubic sequence: a(n) = an³ + bn² + cn + d."""
    if len(seq) < 4:
        return None
    
    d1 = differences(seq)
    d2 = differences(d1)
    d3 = differences(d2)
    
    if not d3:
        return None
    
    if all_close(d3, d3[0]):
        # Third difference = 6a
        a = d3[0] / 6
        
        # Use the values to solve for b, c, d
        # a(1) = a + b + c + d = seq[0]
        # a(2) = 8a + 4b + 2c + d = seq[1]
        # a(3) = 27a + 9b + 3c + d = seq[2]
        # a(4) = 64a + 16b + 4c + d = seq[3]
        
        # From differences:
        # d1[0] = 7a + 3b + c
        # d2[0] = 12a + 2b
        # So b = (d2[0] - 12a) / 2
        b = (d2[0] - 12 * a) / 2
        
        # c = d1[0] - 7a - 3b
        c = d1[0] - 7 * a - 3 * b
        
        # d = seq[0] - a - b - c
        d = seq[0] - a - b - c
        
        # Build equation
        parts = []
        for coef, var in [(a, "n³"), (b, "n²"), (c, "n")]:
            if not is_close(coef, 0):
                if not parts:
                    if is_close(coef, 1):
                        parts.append(var)
                    elif is_close(coef, -1):
                        parts.append(f"-{var}")
                    else:
                        parts.append(f"{format_number(coef)}{var}")
                else:
                    if coef > 0:
                        if is_close(coef, 1):
                            parts.append(f" + {var}")
                        else:
                            parts.append(f" + {format_number(coef)}{var}")
                    else:
                        if is_close(coef, -1):
                            parts.append(f" - {var}")
                        else:
                            parts.append(f" - {format_number(abs(coef))}{var}")
        
        if not is_close(d, 0) or not parts:
            if parts:
                if d > 0:
                    parts.append(f" + {format_number(d)}")
                else:
                    parts.append(f" - {format_number(abs(d))}")
            else:
                parts.append(format_number(d))
        
        eq = f"a(n) = {''.join(parts)}"
        
        return (
            "Cubic",
            eq,
            lambda n, a=a, b=b, c=c, d=d: a * n**3 + b * n**2 + c * n + d
        )
    return None


def detect_polynomial(seq: List[float], max_degree: int = 10) -> Optional[Tuple[str, str, Callable[[int], float]]]:
    """Detect polynomial sequence of degree up to max_degree using finite differences."""
    if len(seq) < 3:
        return None
    
    # Already handled by specific functions for degree <= 3
    diffs = [seq]
    for degree in range(1, min(len(seq), max_degree + 1)):
        new_diff = differences(diffs[-1])
        if not new_diff:
            break
        diffs.append(new_diff)
        
        # Check if this level of differences is constant
        if all_close(new_diff, new_diff[0]):
            if degree <= 3:
                return None  # Let specific detectors handle these
            
            # Reconstruct polynomial coefficients using Newton's forward difference formula
            coefficients = []
            for k in range(degree + 1):
                # Coefficient for n^k term involves binomial coefficients
                coef = diffs[k][0] / math.factorial(k)
                coefficients.append(coef)
            
            # Convert from Newton form to standard form
            # This is complex, so we'll use numerical approach
            def poly_func(n, diffs=diffs, degree=degree):
                result = 0
                for k in range(degree + 1):
                    # Binomial coefficient C(n-1, k)
                    binom = 1
                    for j in range(k):
                        binom *= (n - 1 - j) / (j + 1)
                    result += diffs[k][0] * binom
                return result
            
            eq = f"a(n) = polynomial of degree {degree}"
            
            return (
                f"Polynomial (degree {degree})",
                eq,
                poly_func
            )
    
    return None


def detect_fibonacci_like(seq: List[float]) -> Optional[Tuple[str, str, Callable[[int], float]]]:
    """Detect Fibonacci-like sequence: a(n) = a(n-1) + a(n-2)."""
    if len(seq) < 3:
        return None
    
    # Check if each term is sum of two previous
    for i in range(2, len(seq)):
        if not is_close(seq[i], seq[i-1] + seq[i-2]):
            return None
    
    a, b = seq[0], seq[1]
    
    if is_close(a, 1) and is_close(b, 1):
        seq_name = "Fibonacci"
    elif is_close(a, 2) and is_close(b, 1):
        seq_name = "Lucas"
    else:
        seq_name = "Fibonacci-like"
    
    eq = f"a(n) = a(n-1) + a(n-2), a(1)={format_number(a)}, a(2)={format_number(b)}"
    
    def fib_gen(n, a=a, b=b):
        if n == 1:
            return a
        if n == 2:
            return b
        x, y = a, b
        for _ in range(n - 2):
            x, y = y, x + y
        return y
    
    return (seq_name, eq, fib_gen)


def detect_tribonacci_like(seq: List[float]) -> Optional[Tuple[str, str, Callable[[int], float]]]:
    """Detect Tribonacci-like sequence: a(n) = a(n-1) + a(n-2) + a(n-3)."""
    if len(seq) < 4:
        return None
    
    for i in range(3, len(seq)):
        if not is_close(seq[i], seq[i-1] + seq[i-2] + seq[i-3]):
            return None
    
    a, b, c = seq[0], seq[1], seq[2]
    
    if is_close(a, 0) and is_close(b, 0) and is_close(c, 1):
        seq_name = "Tribonacci"
    elif is_close(a, 1) and is_close(b, 1) and is_close(c, 1):
        seq_name = "Tribonacci (1,1,1 start)"
    else:
        seq_name = "Tribonacci-like"
    
    eq = f"a(n) = a(n-1) + a(n-2) + a(n-3), a(1)={format_number(a)}, a(2)={format_number(b)}, a(3)={format_number(c)}"
    
    def trib_gen(n, a=a, b=b, c=c):
        if n == 1:
            return a
        if n == 2:
            return b
        if n == 3:
            return c
        x, y, z = a, b, c
        for _ in range(n - 3):
            x, y, z = y, z, x + y + z
        return z
    
    return (seq_name, eq, trib_gen)


def detect_factorial(seq: List[float]) -> Optional[Tuple[str, str, Callable[[int], float]]]:
    """Detect factorial sequence or factorial multiples."""
    if len(seq) < 3:
        return None
    
    # Check standard factorial: 1, 2, 6, 24, 120, ...
    factorials = [1]
    for i in range(1, len(seq) + 5):
        factorials.append(factorials[-1] * (i + 1))
    
    # Check if seq matches factorials starting at some point
    for start in range(len(factorials) - len(seq)):
        if all(is_close(seq[i], factorials[start + i]) for i in range(len(seq))):
            offset = start
            eq = f"a(n) = (n + {offset})!" if offset > 0 else "a(n) = n!"
            
            def fact_gen(n, offset=offset):
                return math.factorial(n + offset)
            
            return ("Factorial", eq, fact_gen)
    
    # Check if it's a multiple of factorial
    if seq[0] != 0:
        potential_multiplier = seq[0] / 1  # First term / 1!
        match = True
        for i, val in enumerate(seq):
            expected = potential_multiplier * math.factorial(i + 1)
            if not is_close(val, expected):
                match = False
                break
        
        if match:
            eq = f"a(n) = {format_number(potential_multiplier)} × n!"
            
            def fact_mult_gen(n, m=potential_multiplier):
                return m * math.factorial(n)
            
            return ("Factorial Multiple", eq, fact_mult_gen)
    
    return None


def detect_power_sequence(seq: List[float]) -> Optional[Tuple[str, str, Callable[[int], float]]]:
    """Detect power sequence: a(n) = n^k for some k."""
    if len(seq) < 3:
        return None
    
    # Try different powers
    for k in range(1, 11):
        match = True
        for i, val in enumerate(seq):
            n = i + 1
            if not is_close(val, n ** k):
                match = False
                break
        
        if match:
            if k == 1:
                name = "Natural Numbers"
            elif k == 2:
                name = "Perfect Squares"
            elif k == 3:
                name = "Perfect Cubes"
            else:
                name = f"Powers of n (n^{k})"
            
            eq = f"a(n) = n^{k}" if k > 1 else "a(n) = n"
            
            return (name, eq, lambda n, k=k: n ** k)
    
    return None


def detect_triangular(seq: List[float]) -> Optional[Tuple[str, str, Callable[[int], float]]]:
    """Detect triangular numbers: 1, 3, 6, 10, 15, ..."""
    if len(seq) < 3:
        return None
    
    for i, val in enumerate(seq):
        n = i + 1
        expected = n * (n + 1) // 2
        if not is_close(val, expected):
            return None
    
    return (
        "Triangular Numbers",
        "a(n) = n(n+1)/2",
        lambda n: n * (n + 1) // 2
    )


def detect_pentagonal(seq: List[float]) -> Optional[Tuple[str, str, Callable[[int], float]]]:
    """Detect pentagonal numbers: 1, 5, 12, 22, 35, ..."""
    if len(seq) < 3:
        return None
    
    for i, val in enumerate(seq):
        n = i + 1
        expected = n * (3 * n - 1) // 2
        if not is_close(val, expected):
            return None
    
    return (
        "Pentagonal Numbers",
        "a(n) = n(3n-1)/2",
        lambda n: n * (3 * n - 1) // 2
    )


def detect_hexagonal(seq: List[float]) -> Optional[Tuple[str, str, Callable[[int], float]]]:
    """Detect hexagonal numbers: 1, 6, 15, 28, 45, ..."""
    if len(seq) < 3:
        return None
    
    for i, val in enumerate(seq):
        n = i + 1
        expected = n * (2 * n - 1)
        if not is_close(val, expected):
            return None
    
    return (
        "Hexagonal Numbers",
        "a(n) = n(2n-1)",
        lambda n: n * (2 * n - 1)
    )


def detect_primes(seq: List[float]) -> Optional[Tuple[str, str, Callable[[int], float]]]:
    """Detect prime number sequence."""
    if len(seq) < 3:
        return None
    
    def is_prime(n):
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    def get_primes(count):
        primes = []
        n = 2
        while len(primes) < count:
            if is_prime(n):
                primes.append(n)
            n += 1
        return primes
    
    # Check if all values are integers
    if not all(is_close(v, round(v)) and round(v) > 0 for v in seq):
        return None
    
    int_seq = [int(round(v)) for v in seq]
    
    # Generate enough primes
    primes = get_primes(len(seq) + 10)
    
    # Check if sequence matches primes starting at some index
    for start in range(len(primes) - len(seq)):
        if int_seq == primes[start:start + len(seq)]:
            if start == 0:
                eq = "a(n) = nth prime number"
            else:
                eq = f"a(n) = (n + {start})th prime number"
            
            def prime_gen(n, start=start, get_primes=get_primes):
                return get_primes(n + start + 5)[n + start - 1]
            
            return ("Prime Numbers", eq, prime_gen)
    
    return None


def detect_exponential(seq: List[float]) -> Optional[Tuple[str, str, Callable[[int], float]]]:
    """Detect exponential sequence: a(n) = k * b^n."""
    if len(seq) < 3:
        return None
    
    # Check for zeros
    if any(is_close(x, 0) for x in seq):
        return None
    
    # Check if all same sign
    if not (all(x > 0 for x in seq) or all(x < 0 for x in seq)):
        return None
    
    r_list = ratios(seq)
    if r_list is None:
        return None
    
    r = r_list[0]
    if not all_close(r_list, r):
        return None
    
    # This is geometric, already handled
    # But check for pure exponential: a(n) = b^n
    for base in [2, 3, 5, 10, math.e]:
        match = True
        multiplier = None
        for i, val in enumerate(seq):
            n = i + 1
            expected = base ** n
            if multiplier is None:
                multiplier = val / expected
            if not is_close(val, multiplier * expected):
                match = False
                break
        
        if match:
            if is_close(multiplier, 1):
                if is_close(base, math.e):
                    return ("Exponential", "a(n) = e^n", lambda n: math.e ** n)
                else:
                    return (
                        f"Powers of {int(base)}",
                        f"a(n) = {int(base)}^n",
                        lambda n, b=base: b ** n
                    )
    
    return None


def detect_alternating(seq: List[float]) -> Optional[Tuple[str, str, Callable[[int], float]]]:
    """Detect alternating sequences."""
    if len(seq) < 4:
        return None
    
    # Check for alternating signs
    signs = [1 if x >= 0 else -1 for x in seq]
    alternating_signs = all(signs[i] * signs[i+1] < 0 for i in range(len(signs) - 1))
    
    if not alternating_signs:
        return None
    
    # Get absolute values and check pattern
    abs_seq = [abs(x) for x in seq]
    
    # Check if absolute values form arithmetic sequence
    result = detect_arithmetic(abs_seq)
    if result:
        _, abs_eq, abs_func = result
        
        # Determine sign pattern
        if signs[0] > 0:
            sign_str = "(-1)^(n+1)"
            sign_func = lambda n: 1 if n % 2 == 1 else -1
        else:
            sign_str = "(-1)^n"
            sign_func = lambda n: -1 if n % 2 == 1 else 1
        
        # Extract the expression part
        abs_expr = abs_eq.replace("a(n) = ", "")
        eq = f"a(n) = {sign_str} × ({abs_expr})"
        
        return (
            "Alternating Arithmetic",
            eq,
            lambda n, af=abs_func, sf=sign_func: sf(n) * af(n)
        )
    
    # Check if absolute values form geometric sequence
    result = detect_geometric(abs_seq)
    if result:
        _, abs_eq, abs_func = result
        
        if signs[0] > 0:
            sign_str = "(-1)^(n+1)"
            sign_func = lambda n: 1 if n % 2 == 1 else -1
        else:
            sign_str = "(-1)^n"
            sign_func = lambda n: -1 if n % 2 == 1 else 1
        
        abs_expr = abs_eq.replace("a(n) = ", "")
        eq = f"a(n) = {sign_str} × ({abs_expr})"
        
        return (
            "Alternating Geometric",
            eq,
            lambda n, af=abs_func, sf=sign_func: sf(n) * af(n)
        )
    
    return None


def detect_sum_of_n(seq: List[float]) -> Optional[Tuple[str, str, Callable[[int], float]]]:
    """Detect sum sequences: sum of first n integers, squares, cubes, etc."""
    if len(seq) < 3:
        return None
    
    # Sum of first n natural numbers: n(n+1)/2
    match = True
    for i, val in enumerate(seq):
        n = i + 1
        expected = n * (n + 1) // 2
        if not is_close(val, expected):
            match = False
            break
    if match:
        return (
            "Sum of Natural Numbers",
            "a(n) = n(n+1)/2 = 1+2+...+n",
            lambda n: n * (n + 1) // 2
        )
    
    # Sum of squares: n(n+1)(2n+1)/6
    match = True
    for i, val in enumerate(seq):
        n = i + 1
        expected = n * (n + 1) * (2 * n + 1) // 6
        if not is_close(val, expected):
            match = False
            break
    if match:
        return (
            "Sum of Squares",
            "a(n) = n(n+1)(2n+1)/6 = 1²+2²+...+n²",
            lambda n: n * (n + 1) * (2 * n + 1) // 6
        )
    
    # Sum of cubes: [n(n+1)/2]²
    match = True
    for i, val in enumerate(seq):
        n = i + 1
        expected = (n * (n + 1) // 2) ** 2
        if not is_close(val, expected):
            match = False
            break
    if match:
        return (
            "Sum of Cubes",
            "a(n) = [n(n+1)/2]² = 1³+2³+...+n³",
            lambda n: (n * (n + 1) // 2) ** 2
        )
    
    return None


def detect_catalan(seq: List[float]) -> Optional[Tuple[str, str, Callable[[int], float]]]:
    """Detect Catalan numbers: 1, 1, 2, 5, 14, 42, ..."""
    if len(seq) < 3:
        return None
    
    def catalan(n):
        if n <= 1:
            return 1
        return math.comb(2 * n, n) // (n + 1)
    
    # Check starting from index 0
    for offset in range(3):
        match = True
        for i, val in enumerate(seq):
            n = i + offset
            if not is_close(val, catalan(n)):
                match = False
                break
        if match:
            eq = f"a(n) = C(2n,n)/(n+1)" if offset == 0 else f"a(n) = C(2(n+{offset-1}),n+{offset-1})/(n+{offset})"
            return (
                "Catalan Numbers",
                eq,
                lambda n, o=offset: catalan(n + o - 1)
            )
    
    return None


def detect_bell(seq: List[float]) -> Optional[Tuple[str, str, Callable[[int], float]]]:
    """Detect Bell numbers: 1, 1, 2, 5, 15, 52, ..."""
    if len(seq) < 3:
        return None
    
    def bell(n):
        # Using Bell triangle
        if n == 0:
            return 1
        bell_tri = [[1]]
        for i in range(1, n + 1):
            row = [bell_tri[i-1][-1]]
            for j in range(1, i + 1):
                row.append(row[j-1] + bell_tri[i-1][j-1])
            bell_tri.append(row)
        return bell_tri[n][0]
    
    for offset in range(3):
        match = True
        for i, val in enumerate(seq):
            n = i + offset
            if not is_close(val, bell(n)):
                match = False
                break
        if match:
            eq = "a(n) = B(n) (Bell number)"
            return (
                "Bell Numbers",
                eq,
                lambda n, o=offset: bell(n + o - 1)
            )
    
    return None


def detect_double_factorial(seq: List[float]) -> Optional[Tuple[str, str, Callable[[int], float]]]:
    """Detect double factorial sequences."""
    if len(seq) < 3:
        return None
    
    def double_factorial(n):
        if n <= 0:
            return 1
        result = 1
        while n > 0:
            result *= n
            n -= 2
        return result
    
    # Odd double factorial: 1, 3, 15, 105, ...
    match = True
    for i, val in enumerate(seq):
        n = 2 * i + 1
        if not is_close(val, double_factorial(n)):
            match = False
            break
    if match:
        return (
            "Odd Double Factorial",
            "a(n) = (2n-1)!! = 1×3×5×...×(2n-1)",
            lambda n: double_factorial(2 * n - 1)
        )
    
    # Even double factorial: 2, 8, 48, 384, ...
    match = True
    for i, val in enumerate(seq):
        n = 2 * (i + 1)
        if not is_close(val, double_factorial(n)):
            match = False
            break
    if match:
        return (
            "Even Double Factorial",
            "a(n) = (2n)!! = 2×4×6×...×(2n)",
            lambda n: double_factorial(2 * n)
        )
    
    return None


def detect_look_and_say(seq: List[float]) -> Optional[Tuple[str, str, Callable[[int], float]]]:
    """Detect Look-and-Say sequence: 1, 11, 21, 1211, 111221, ..."""
    if len(seq) < 3:
        return None
    
    # Check if all are positive integers
    if not all(is_close(v, round(v)) and round(v) > 0 for v in seq):
        return None
    
    int_seq = [int(round(v)) for v in seq]
    
    def next_look_and_say(n):
        s = str(n)
        result = []
        i = 0
        while i < len(s):
            digit = s[i]
            count = 1
            while i + count < len(s) and s[i + count] == digit:
                count += 1
            result.append(str(count) + digit)
            i += count
        return int(''.join(result))
    
    # Verify the sequence follows look-and-say rule
    for i in range(len(int_seq) - 1):
        if next_look_and_say(int_seq[i]) != int_seq[i + 1]:
            return None
    
    def generate_las(n, start=int_seq[0]):
        if n == 1:
            return start
        val = start
        for _ in range(n - 1):
            val = next_look_and_say(val)
        return val
    
    return (
        "Look-and-Say",
        f"a(n) = Look-and-Say sequence starting with {int_seq[0]}",
        lambda n, s=int_seq[0]: generate_las(n, s)
    )


def detect_recurrence(seq: List[float]) -> Optional[Tuple[str, str, Callable[[int], float]]]:
    """Detect linear recurrence relations: a(n) = c1*a(n-1) + c2*a(n-2) + ..."""
    if len(seq) < 4:
        return None
    
    # Try different orders of recurrence
    for order in range(2, min(len(seq) // 2, 5)):
        # Set up system of equations
        # a(order+1) = c1*a(order) + c2*a(order-1) + ... + c_order*a(1)
        
        # Build matrix A and vector b
        n_equations = len(seq) - order
        if n_equations < order:
            continue
        
        # Solve using least squares approach
        A = []
        b = []
        for i in range(order, len(seq)):
            row = [seq[i - j - 1] for j in range(order)]
            A.append(row)
            b.append(seq[i])
        
        # Simple Gaussian elimination for small systems
        try:
            # Use numpy-like approach with pure Python
            coeffs = solve_linear_system(A[:order], b[:order])
            if coeffs is None:
                continue
            
            # Verify with remaining equations
            valid = True
            for i in range(order, n_equations):
                predicted = sum(coeffs[j] * A[i][j] for j in range(order))
                if not is_close(predicted, b[i]):
                    valid = False
                    break
            
            if valid:
                # Build equation string
                terms = []
                for j, c in enumerate(coeffs):
                    if not is_close(c, 0):
                        coef_str = format_number(c)
                        if is_close(c, 1):
                            terms.append(f"a(n-{j+1})")
                        elif is_close(c, -1):
                            terms.append(f"-a(n-{j+1})")
                        else:
                            terms.append(f"{coef_str}×a(n-{j+1})")
                
                eq = "a(n) = " + " + ".join(terms).replace("+ -", "- ")
                
                def recurrence_gen(n, seq=seq[:], coeffs=coeffs, order=order):
                    # Build sequence up to n
                    s = list(seq)
                    while len(s) < n:
                        next_val = sum(coeffs[j] * s[-(j+1)] for j in range(order))
                        s.append(next_val)
                    return s[n - 1]
                
                return (
                    f"Linear Recurrence (order {order})",
                    eq,
                    recurrence_gen
                )
        except Exception:
            continue
    
    return None


def solve_linear_system(A: List[List[float]], b: List[float]) -> Optional[List[float]]:
    """Solve a linear system Ax = b using Gaussian elimination."""
    n = len(A)
    if n == 0 or len(A[0]) != n or len(b) != n:
        return None
    
    # Augmented matrix
    aug = [row[:] + [b[i]] for i, row in enumerate(A)]
    
    # Forward elimination
    for col in range(n):
        # Find pivot
        max_row = col
        for row in range(col + 1, n):
            if abs(aug[row][col]) > abs(aug[max_row][col]):
                max_row = row
        aug[col], aug[max_row] = aug[max_row], aug[col]
        
        if is_close(aug[col][col], 0):
            return None
        
        # Eliminate
        for row in range(col + 1, n):
            factor = aug[row][col] / aug[col][col]
            for j in range(col, n + 1):
                aug[row][j] -= factor * aug[col][j]
    
    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = aug[i][n]
        for j in range(i + 1, n):
            x[i] -= aug[i][j] * x[j]
        x[i] /= aug[i][i]
    
    return x


def detect_harmonic_like(seq: List[float]) -> Optional[Tuple[str, str, Callable[[int], float]]]:
    """Detect sequences based on 1/n patterns."""
    if len(seq) < 3:
        return None
    
    # Check for 1/n
    match = True
    for i, val in enumerate(seq):
        n = i + 1
        if not is_close(val, 1 / n):
            match = False
            break
    if match:
        return (
            "Harmonic",
            "a(n) = 1/n",
            lambda n: 1 / n
        )
    
    # Check for k/n
    if seq[0] != 0:
        k = seq[0]  # k/1 = k
        match = True
        for i, val in enumerate(seq):
            n = i + 1
            if not is_close(val, k / n):
                match = False
                break
        if match:
            return (
                "Harmonic Multiple",
                f"a(n) = {format_number(k)}/n",
                lambda n, k=k: k / n
            )
    
    # Check for 1/n^2
    match = True
    for i, val in enumerate(seq):
        n = i + 1
        if not is_close(val, 1 / (n * n)):
            match = False
            break
    if match:
        return (
            "Inverse Square",
            "a(n) = 1/n²",
            lambda n: 1 / (n * n)
        )
    
    return None


def detect_repunits(seq: List[float]) -> Optional[Tuple[str, str, Callable[[int], float]]]:
    """Detect repunit sequences: 1, 11, 111, 1111, ..."""
    if len(seq) < 3:
        return None
    
    # Check if all are positive integers
    if not all(is_close(v, round(v)) and round(v) > 0 for v in seq):
        return None
    
    int_seq = [int(round(v)) for v in seq]
    
    # Check standard repunits
    match = True
    for i, val in enumerate(int_seq):
        n = i + 1
        expected = int('1' * n)
        if val != expected:
            match = False
            break
    if match:
        return (
            "Repunits",
            "a(n) = (10^n - 1)/9 = 111...1 (n ones)",
            lambda n: (10 ** n - 1) // 9
        )
    
    return None


def detect_square_pyramidal(seq: List[float]) -> Optional[Tuple[str, str, Callable[[int], float]]]:
    """Detect square pyramidal numbers: 1, 5, 14, 30, 55, ..."""
    if len(seq) < 3:
        return None
    
    for i, val in enumerate(seq):
        n = i + 1
        expected = n * (n + 1) * (2 * n + 1) // 6
        if not is_close(val, expected):
            return None
    
    return (
        "Square Pyramidal",
        "a(n) = n(n+1)(2n+1)/6",
        lambda n: n * (n + 1) * (2 * n + 1) // 6
    )


def detect_tetrahedral(seq: List[float]) -> Optional[Tuple[str, str, Callable[[int], float]]]:
    """Detect tetrahedral numbers: 1, 4, 10, 20, 35, ..."""
    if len(seq) < 3:
        return None
    
    for i, val in enumerate(seq):
        n = i + 1
        expected = n * (n + 1) * (n + 2) // 6
        if not is_close(val, expected):
            return None
    
    return (
        "Tetrahedral Numbers",
        "a(n) = n(n+1)(n+2)/6",
        lambda n: n * (n + 1) * (n + 2) // 6
    )


def detect_centered_polygonal(seq: List[float]) -> Optional[Tuple[str, str, Callable[[int], float]]]:
    """Detect centered polygonal number sequences."""
    if len(seq) < 3:
        return None
    
    # Centered triangular: 1, 4, 10, 19, 31, ...
    # Formula: (3n² - 3n + 2) / 2
    match = True
    for i, val in enumerate(seq):
        n = i + 1
        expected = (3 * n * n - 3 * n + 2) // 2
        if not is_close(val, expected):
            match = False
            break
    if match:
        return (
            "Centered Triangular",
            "a(n) = (3n² - 3n + 2)/2",
            lambda n: (3 * n * n - 3 * n + 2) // 2
        )
    
    # Centered square: 1, 5, 13, 25, 41, ...
    # Formula: 2n² - 2n + 1
    match = True
    for i, val in enumerate(seq):
        n = i + 1
        expected = 2 * n * n - 2 * n + 1
        if not is_close(val, expected):
            match = False
            break
    if match:
        return (
            "Centered Square",
            "a(n) = 2n² - 2n + 1",
            lambda n: 2 * n * n - 2 * n + 1
        )
    
    # Centered hexagonal: 1, 7, 19, 37, 61, ...
    # Formula: 3n² - 3n + 1
    match = True
    for i, val in enumerate(seq):
        n = i + 1
        expected = 3 * n * n - 3 * n + 1
        if not is_close(val, expected):
            match = False
            break
    if match:
        return (
            "Centered Hexagonal",
            "a(n) = 3n² - 3n + 1",
            lambda n: 3 * n * n - 3 * n + 1
        )
    
    return None


def detect_perfect_numbers(seq: List[float]) -> Optional[Tuple[str, str, Callable[[int], float]]]:
    """Detect perfect numbers: 6, 28, 496, 8128, ..."""
    if len(seq) < 2:
        return None
    
    perfects = [6, 28, 496, 8128, 33550336]
    
    # Check if sequence is a prefix of perfect numbers
    if not all(is_close(v, round(v)) and round(v) > 0 for v in seq):
        return None
    
    int_seq = [int(round(v)) for v in seq]
    
    for start in range(len(perfects) - len(seq) + 1):
        if int_seq == perfects[start:start + len(seq)]:
            def perfect_gen(n, start=start):
                p_list = [6, 28, 496, 8128, 33550336, 8589869056, 137438691328]
                idx = n + start - 1
                if idx < len(p_list):
                    return p_list[idx]
                return None  # Beyond known perfect numbers
            
            return (
                "Perfect Numbers",
                "a(n) = nth perfect number",
                perfect_gen
            )
    
    return None


# =============================================================================
# Main Detection Engine
# =============================================================================

def analyze_sequence(seq: List[float]) -> Optional[Tuple[str, str, Callable[[int], float]]]:
    """Analyze a sequence and return its type, equation, and generator function."""
    
    # List of detectors in order of specificity (most specific first)
    detectors = [
        detect_constant,
        detect_look_and_say,
        detect_primes,
        detect_perfect_numbers,
        detect_catalan,
        detect_bell,
        detect_factorial,
        detect_double_factorial,
        detect_tribonacci_like,
        detect_fibonacci_like,
        detect_repunits,
        detect_power_sequence,
        detect_triangular,
        detect_pentagonal,
        detect_hexagonal,
        detect_tetrahedral,
        detect_square_pyramidal,
        detect_centered_polygonal,
        detect_sum_of_n,
        detect_harmonic_like,
        detect_alternating,
        detect_exponential,
        detect_arithmetic,
        detect_geometric,
        detect_quadratic,
        detect_cubic,
        detect_recurrence,
        detect_polynomial,
    ]
    
    for detector in detectors:
        try:
            result = detector(seq)
            if result:
                return result
        except Exception:
            continue
    
    return None


def generate_next_terms(generator: Callable[[int], float], start_n: int, count: int) -> List[float]:
    """Generate the next 'count' terms starting from position 'start_n'."""
    terms = []
    for i in range(count):
        try:
            term = generator(start_n + i)
            if term is None:
                break
            terms.append(term)
        except Exception:
            break
    return terms


def main():
    """Main function to run the sequence predictor."""
    print("=" * 60)
    print("       MATHEMATICAL SEQUENCE PREDICTOR")
    print("=" * 60)
    print()
    
    while True:
        try:
            print("Enter at least 3 numbers separated by comma or space")
            print("(or type 'quit' to exit):")
            user_input = input("> ").strip()
            t1 = time.perf_counter()
            if user_input.lower() in ('quit', 'exit', 'q'):
                print("\nGoodbye!")
                break
            
            if not user_input:
                print("\nError: Please enter some numbers.\n")
                continue
            
            # Parse input
            try:
                seq = parse_input(user_input)
            except ValueError as e:
                print(f"\nError: {e}\n")
                continue
            
            if len(seq) < 3:
                print("\nError: Please enter at least 3 numbers.\n")
                continue
            
            print()
            print("-" * 60)
            print(f"Input sequence: {', '.join(format_number(x) for x in seq)}")
            print("-" * 60)
            
            # Analyze the sequence
            result = analyze_sequence(seq)
            
            if result:
                seq_type, equation, generator = result
                
                # Generate next 5 terms
                next_terms = generate_next_terms(generator, len(seq) + 1, 5)
                
                print(f"\n  Type:     {seq_type}")
                print(f"  Equation: {equation}")
                
                if next_terms:
                    formatted_terms = '  '.join(format_number(t) for t in next_terms)
                    print(f"  Next 5:   {formatted_terms}")
                else:
                    print("  Next 5:   (Unable to generate further terms)")
                
                # Verification: show reconstructed sequence
                reconstructed = [generator(i + 1) for i in range(len(seq))]
                if all(is_close(seq[i], reconstructed[i]) for i in range(len(seq))):
                    print("\n  ✓ Pattern verified against input sequence")
                else:
                    print("\n  ⚠ Note: Pattern is approximate")
            else:
                print("\n  Type:     Unknown")
                print("  Equation: Unable to determine pattern")
                print("  Next 5:   (Unable to predict)")
                print("\n  Tip: Try providing more terms for better pattern detection.")
            
            print("-" * 60)
            t2 = time.perf_counter()
            print(t2-t1)
            print()
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except EOFError:
            print("\n\nEnd of input. Goodbye!")
            break


if __name__ == "__main__":
    main()