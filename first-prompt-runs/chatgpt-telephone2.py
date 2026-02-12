from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Union


TextLike = Union[str, bytes]


@dataclass(frozen=True)
class LossyHalfCodec:
    """
    A simple *lossy* "halve-then-reconstruct" codec.

    Compression rule (guaranteed ~half size):
      - Keep every second symbol (even indices: 0, 2, 4, ...).
      - Store the original length so decompression can restore the exact length.

    Decompression rule (best-effort reconstruction):
      - Expand by inserting a predicted symbol between each kept symbol.
      - The predictor is context-based and deterministic:
          * If neighbors are equal -> repeat that symbol.
          * Else -> choose the "median" of the two (for bytes: numeric median,
                   for str: median of Unicode code points).
      - If the original length was odd, the final kept symbol is appended.

    Properties:
      - Compression always produces ceil(n/2) symbols, i.e. about half.
      - Decompression always returns exactly the original length.
      - Robust to empty inputs, length 1, and arbitrary bytes/unicode text.
      - Fast: O(n) time, O(n) memory.

    Notes:
      - Because compression discards information, decompression cannot perfectly
        reproduce arbitrary inputs. It is a best-effort approximation.
    """

    @staticmethod
    def compress(message: TextLike) -> "CompressedMessage":
        if not isinstance(message, (str, bytes)):
            raise TypeError("message must be of type 'str' or 'bytes'")

        n = len(message)
        if n == 0:
            return CompressedMessage(payload=message, original_length=0)

        # Keep even indices: 0, 2, 4, ...
        payload = message[0:n:2]
        return CompressedMessage(payload=payload, original_length=n)

    @staticmethod
    def decompress(blob: "CompressedMessage") -> TextLike:
        if not isinstance(blob, CompressedMessage):
            raise TypeError("blob must be a CompressedMessage instance")

        payload = blob.payload
        n = blob.original_length

        if not isinstance(payload, (str, bytes)):
            raise TypeError("CompressedMessage.payload must be 'str' or 'bytes'")
        if n < 0:
            raise ValueError("original_length must be non-negative")
        if n == 0:
            # For empty original, ignore payload content; return correct type.
            return payload[:0]

        # The payload must have exactly ceil(n/2) symbols for this codec.
        expected = (n + 1) // 2
        if len(payload) != expected:
            raise ValueError(
                f"Invalid payload length: got {len(payload)}, expected {expected} "
                f"for original_length={n}"
            )

        # Fast path: n == 1
        if n == 1:
            return payload[:1]

        if isinstance(payload, bytes):
            return _decompress_bytes(payload, n)
        else:
            return _decompress_str(payload, n)


@dataclass(frozen=True)
class CompressedMessage:
    payload: TextLike
    original_length: int


def _predict_between_bytes(left: int, right: int) -> int:
    if left == right:
        return left
    # Numeric median (integer) for stability.
    # Example: left=65, right=67 -> 66
    return (left + right) // 2


def _predict_between_str(left: str, right: str) -> str:
    if left == right:
        return left
    # Median by Unicode code point (deterministic and cheap).
    return chr((ord(left) + ord(right)) // 2)


def _decompress_bytes(payload: bytes, original_length: int) -> bytes:
    # Build a bytearray for speed.
    out = bytearray(original_length)

    # Place kept symbols to even indices.
    out[0:original_length:2] = payload

    # Fill odd indices using neighbor predictions.
    # Odd index i has neighbors i-1 and i+1, except at the end.
    last_index = original_length - 1
    for i in range(1, original_length, 2):
        left = out[i - 1]
        if i + 1 <= last_index:
            right = out[i + 1]
        else:
            # If there is no right neighbor (happens only when original_length is even),
            # fall back to repeating the left neighbor.
            right = left
        out[i] = _predict_between_bytes(left, right)

    return bytes(out)


def _decompress_str(payload: str, original_length: int) -> str:
    # Strings are immutable; build a list of chars for speed.
    out: list[str] = [""] * original_length

    # Place kept symbols to even indices.
    out[0:original_length:2] = list(payload)

    # Fill odd indices using neighbor predictions.
    last_index = original_length - 1
    for i in range(1, original_length, 2):
        left = out[i - 1]
        if i + 1 <= last_index:
            right = out[i + 1]
        else:
            right = left
        out[i] = _predict_between_str(left, right)

    return "".join(out)


# ----------------------------
# Example usage + small tests
# ----------------------------

def _roundtrip_demo(message: TextLike) -> None:
    blob = LossyHalfCodec.compress(message)
    recovered = LossyHalfCodec.decompress(blob)
    print("original:  ", message)
    print("compressed:", blob.payload, f"(orig_len={blob.original_length})")
    print("recovered: ", recovered)
    print("sizes:     ", len(message), "->", len(blob.payload), "->", len(recovered))
    print()


def _self_test() -> None:
    # Edge cases
    for m in ["", "A", "AB", "ABC", "ABCD", "😀", "😀😃😄😁", "aaaaaa", "abababab"]:
        blob = LossyHalfCodec.compress(m)
        rec = LossyHalfCodec.decompress(blob)
        assert len(rec) == len(m)
        assert isinstance(rec, str)

    for m in [b"", b"\x00", b"\x00\xff", b"hello", bytes(range(1, 32))]:
        blob = LossyHalfCodec.compress(m)
        rec = LossyHalfCodec.decompress(blob)
        assert len(rec) == len(m)
        assert isinstance(rec, bytes)

    # Robustness: invalid blob
    try:
        LossyHalfCodec.decompress(CompressedMessage(payload="abc", original_length=10))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for invalid payload length")

    print("Self-test passed.")


if __name__ == "__main__":
    _self_test()
    _roundtrip_demo("")
