#!/usr/bin/env python3
"""
Hybrid compression/decompression demo.

Goal:
- Try high-accuracy (lossless) compression first (zlib + base64).
- If that is not strictly smaller than the original, fall back to a guaranteed-smaller
  lossy compression (drop the last character).
- Always print:
  - the compressed message (printable text)
  - compressed size vs original size
  - decompression accuracy (SequenceMatcher ratio)

Notes:
- For an arbitrary input, it is impossible to guarantee "always smaller" with purely
  lossless compression. This program guarantees "always smaller" for any NON-EMPTY input
  by using a lossy fallback when necessary.
"""

from __future__ import annotations

import base64
import difflib
import zlib
import time


def byte_len(s: str) -> int:
    """Size in bytes when encoded as UTF-8."""
    return len(s.encode("utf-8"))


def lossless_compress_to_printable(s: str, level: int = 9) -> str:
    """
    Lossless compression: UTF-8 -> zlib -> base64 (ascii).
    Returns a printable ASCII string with a prefix marker.
    """
    raw = s.encode("utf-8")
    comp = zlib.compress(raw, level=level)
    b64 = base64.urlsafe_b64encode(comp).decode("ascii")
    return "Z:" + b64


def lossless_decompress_from_printable(payload: str) -> str:
    """Inverse of lossless_compress_to_printable."""
    if not payload.startswith("Z:"):
        raise ValueError("Not a lossless payload (missing 'Z:' prefix).")
    b64 = payload[2:]
    comp = base64.urlsafe_b64decode(b64.encode("ascii"))
    raw = zlib.decompress(comp)
    return raw.decode("utf-8")


def lossy_compress_guaranteed_smaller(s: str) -> str:
    """
    Guaranteed smaller for any non-empty string:
    - Drop the last character.
    This is intentionally simple and robust; it guarantees strict size reduction in chars.
    """
    # Precondition: s is non-empty
    return s[:-1]


def lossy_decompress_best_effort(payload: str) -> str:
    """
    Best-effort decompression for the lossy fallback.
    Since we removed information, we return the payload as-is.
    """
    return payload


def accuracy_ratio(original: str, reconstructed: str) -> float:
    """
    Similarity ratio in [0, 1] using difflib (character-based).
    1.0 means exact match.
    """
    return difflib.SequenceMatcher(a=original, b=reconstructed).ratio()


def compress_then_decompress(message: str) -> tuple[str, str, str]:
    """
    Returns: (method, compressed_message, decompressed_message)
    Method is "lossless" or "lossy".
    """
    if message == "":
        # A strict "compressed must always be smaller" requirement cannot be satisfied for empty input.
        # We handle it explicitly and keep behavior predictable.
        return ("uncompressible_empty", "", "")

    # 1) Try lossless
    try:
        candidate = lossless_compress_to_printable(message, level=9)
        if byte_len(candidate) < byte_len(message):
            recovered = lossless_decompress_from_printable(candidate)
            return ("lossless", candidate, recovered)
    except Exception:
        # If anything goes wrong, fall back to the guaranteed approach.
        pass

    # 2) Guaranteed-smaller lossy fallback
    compressed = lossy_compress_guaranteed_smaller(message)
    recovered = lossy_decompress_best_effort(compressed)
    return ("lossy", compressed, recovered)


def main() -> None:
    message = input("Enter message to compress: ")
    t1 = time.perf_counter()
    method, compressed, decompressed = compress_then_decompress(message)

    if method == "uncompressible_empty":
        print("\nResult:")
        print("  Input was empty. It is impossible to produce a strictly smaller message than empty.")
        print("  Compressed message: ''")
        print("  Sizes (bytes): original=0, compressed=0")
        print("  Decompression accuracy: 1.0000")
        return

    orig_bytes = byte_len(message)
    comp_bytes = byte_len(compressed)

    # Safety check for the strict requirement (for non-empty input).
    # If this triggers, there is a bug in the logic.
    if comp_bytes >= orig_bytes:
        raise RuntimeError(
            f"Compression requirement violated: compressed ({comp_bytes}) "
            f"is not smaller than original ({orig_bytes})."
        )

    acc = accuracy_ratio(message, decompressed)
    ratio = comp_bytes / orig_bytes if orig_bytes else 0.0

    print("\nResult:")
    print(f"  Method: {method}")
    print(f"  Compressed message: {compressed!r}")
    print(f"  Sizes (bytes): original={orig_bytes}, compressed={comp_bytes} "
          f"(compressed/original={ratio:.4f})")
    print(f"  Decompressed message: {decompressed!r}")
    print(f"  Decompression accuracy: {acc:.4f}")
    t2 = time.perf_counter()
    print(t2-t1)


if __name__ == "__main__":
    
    main()
    