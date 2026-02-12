from __future__ import annotations

import math
import time


def count_primes_above_10k(time_limit_s: float = 1.0, start: int = 10_001) -> int:
    """
    Counts how many primes > 10_000 can be generated within `time_limit_s` seconds.

    Approach:
    - Keep a growing list of primes.
    - Test each odd candidate by trial division using primes up to sqrt(n).
    - Use a monotonic clock and stop strictly by elapsed time.

    Robustness / edge cases:
    - Handles non-positive or non-finite time limits by returning 0.
    - Ensures the start is adjusted to the next odd integer > 10_000.
    - Avoids float sqrt inaccuracies by using math.isqrt.
    """
    # Validate time limit
    if not isinstance(time_limit_s, (int, float)) or not math.isfinite(time_limit_s) or time_limit_s <= 0:
        return 0

    # Ensure start is an integer
    try:
        start = int(start)
    except Exception:
        start = 10_001

    # We only count primes strictly greater than 10_000
    if start <= 10_000:
        start = 10_001

    # Quick-start prime list with small primes; enough to trial-divide early candidates.
    primes: list[int] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    # Ensure candidate is odd
    candidate = start if (start & 1) == 1 else start + 1

    deadline = time.perf_counter() + float(time_limit_s)
    count = 0

    # Main timed loop
    while True:
        # Check time frequently and cheaply
        if time.perf_counter() >= deadline:
            break

        # Primality test by dividing only by known primes up to sqrt(candidate)
        limit = math.isqrt(candidate)
        is_prime = True
        for p in primes:
            if p > limit:
                break
            if candidate % p == 0:
                is_prime = False
                break

        if is_prime:
            primes.append(candidate)
            if candidate > 10_000:
                count += 1

        candidate += 2  # skip even numbers

    return count


def main() -> None:
    # Per your requirement: run for ~1 second and output just the count.
    print(count_primes_above_10k(time_limit_s=1.0, start=10_001))


if __name__ == "__main__":
    main()
