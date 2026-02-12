#!/usr/bin/env python3
"""
Matrix-algebra-based lossy compression/decompression demo.

Idea (Walsh–Hadamard transform):
- Treat the message bytes as a vector x.
- Apply an (implicitly defined) orthogonal matrix H (Hadamard): y = Hx.
- Keep only half of the transform coefficients (compression by 2× in coefficient count).
- Reconstruct by padding missing coefficients with zeros and applying the inverse transform.

This is a *lossy* scheme in general: dropping half the coefficients loses information.
The program prints decompression accuracy (exact byte-match rate and MSE).

Notes:
- "Halve the size" is implemented as halving the number of stored coefficients
  (N/2 vs N). Actual byte size depends on how you serialize integers.
- Uses fast O(N log N) Walsh–Hadamard transform (FWHT), not an explicit matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional
import math
import numpy as np


def _next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _fwht_inplace(a: np.ndarray) -> None:
    """
    In-place Fast Walsh–Hadamard Transform (FWHT) for 1D array length power-of-two.

    This computes the unnormalized transform:
        for Hadamard matrix H with entries ±1, y = Hx
    The inverse is the same transform followed by division by N.

    Requirements:
    - a is 1D, contiguous, dtype int64/float64 recommended for headroom.
    - len(a) is a power of two.
    """
    n = a.shape[0]
    if n == 0:
        return
    if (n & (n - 1)) != 0:
        raise ValueError("FWHT requires length to be a power of two.")

    h = 1
    while h < n:
        # Vectorized butterfly operations over blocks of size 2h
        step = h * 2
        for i in range(0, n, step):
            x = a[i : i + h].copy()
            y = a[i + h : i + step].copy()
            a[i : i + h] = x + y
            a[i + h : i + step] = x - y
        h = step


@dataclass(frozen=True)
class CompressedMessage:
    """
    Stores the compressed representation.
    - coeffs: first half of the Hadamard coefficients
    - original_len: original byte length
    - padded_len: transform length (power of two)
    """
    coeffs: np.ndarray
    original_len: int
    padded_len: int


class HadamardCompressor:
    """
    Compress/decompress using a Hadamard (Walsh) transform and coefficient truncation.
    """

    def __init__(self) -> None:
        pass

    def compress_bytes(self, data: bytes) -> CompressedMessage:
        """
        Compress by keeping exactly half of the transform coefficients.

        Edge cases:
        - Empty input -> empty compressed coeffs with padded_len=1.
        """
        original_len = len(data)
        padded_len = _next_power_of_two(max(1, original_len))

        # Convert to signed values centered around 0 to reduce bias.
        x_u8 = np.frombuffer(data, dtype=np.uint8)
        x = np.zeros(padded_len, dtype=np.int64)
        if original_len > 0:
            x[:original_len] = x_u8.astype(np.int64) - 128

        # Transform y = Hx
        _fwht_inplace(x)
        y = x  # transformed in place

        # Keep the first half coefficients (deterministic, no index overhead).
        half = padded_len // 2
        coeffs = y[:half].copy()

        return CompressedMessage(coeffs=coeffs, original_len=original_len, padded_len=padded_len)

    def decompress_bytes(self, cm: CompressedMessage) -> bytes:
        """
        Decompress by padding missing coefficients with zeros and applying inverse transform.

        Inverse for unnormalized FWHT:
            x = (1/N) * H y
        """
        if cm.padded_len <= 0 or (cm.padded_len & (cm.padded_len - 1)) != 0:
            raise ValueError("Invalid padded_len in compressed message (must be power of two).")
        if cm.original_len < 0 or cm.original_len > cm.padded_len:
            raise ValueError("Invalid original_len in compressed message.")

        n = cm.padded_len
        half = n // 2
        if cm.coeffs.shape[0] != half:
            raise ValueError("Invalid coeffs length: expected exactly padded_len//2.")

        # Reconstruct full coefficient vector.
        y = np.zeros(n, dtype=np.int64)
        y[:half] = cm.coeffs.astype(np.int64)

        # Inverse transform: x = (1/n) * H y
        _fwht_inplace(y)
        x_rec = y.astype(np.float64) / float(n)

        # Undo centering and quantize back to bytes
        x_rec = np.rint(x_rec + 128.0)
        x_rec = np.clip(x_rec, 0, 255).astype(np.uint8)

        return x_rec[: cm.original_len].tobytes()

    @staticmethod
    def accuracy_metrics(original: bytes, reconstructed: bytes) -> dict:
        """
        Compute robust, simple accuracy metrics.
        """
        o = np.frombuffer(original, dtype=np.uint8)
        r = np.frombuffer(reconstructed, dtype=np.uint8)

        if o.shape != r.shape:
            # If lengths differ, compare over overlap and penalize missing/excess as mismatches.
            m = min(o.size, r.size)
            exact_overlap = float(np.mean(o[:m] == r[:m])) if m > 0 else 1.0
            exact_full = (exact_overlap * m) / max(o.size, r.size) if max(o.size, r.size) > 0 else 1.0
            # MSE over overlap only (full-length MSE is not well-defined if sizes differ)
            mse = float(np.mean((o[:m].astype(np.float64) - r[:m].astype(np.float64)) ** 2)) if m > 0 else 0.0
        else:
            exact_full = float(np.mean(o == r)) if o.size > 0 else 1.0
            mse = float(np.mean((o.astype(np.float64) - r.astype(np.float64)) ** 2)) if o.size > 0 else 0.0

        return {
            "exact_byte_match_rate": exact_full,  # in [0,1]
            "mse": mse,
            "original_len_bytes": int(len(original)),
            "reconstructed_len_bytes": int(len(reconstructed)),
        }


def demo(message: str) -> None:
    compressor = HadamardCompressor()

    original_bytes = message.encode("utf-8")
    cm = compressor.compress_bytes(original_bytes)
    reconstructed_bytes = compressor.decompress_bytes(cm)

    metrics = compressor.accuracy_metrics(original_bytes, reconstructed_bytes)

    # Try to decode reconstructed text (may be imperfect).
    reconstructed_text = reconstructed_bytes.decode("utf-8", errors="replace")

    print("=== Hadamard 2× (lossy) compression demo ===")
    print(f"Original message (text): {message!r}")
    print(f"Original length (bytes): {metrics['original_len_bytes']}")
    print(f"Padded length for transform (bytes): {cm.padded_len}")
    print(f"Stored coefficients: {cm.coeffs.shape[0]} (exactly half of {cm.padded_len})")

    # Accuracy report
    print("\n--- Decompression accuracy ---")
    print(f"Exact byte match rate: {metrics['exact_byte_match_rate']*100:.2f}%")
    print(f"MSE (bytes): {metrics['mse']:.4f}")

    print("\nReconstructed message (text, best effort):")
    print(reconstructed_text)


if __name__ == "__main__":
    # Example usage. Replace with any input you want.
    demo("Foxes are small wild animals found in many parts of the world. They live in forests, fields, mountains, and even near cities. Foxes have soft fur, long tails, and sharp ears. Most foxes are red, but some are gray, white, or brown. They are clever and careful animals. Foxes usually hunt at night. They eat mice, birds, insects, fruit, and plants. Foxes live alone or in small families. They use dens in the ground for rest and safety. People often see foxes as symbols of intelligence and adaptability. Their survival skills help them live in changing environments around humans today.")
