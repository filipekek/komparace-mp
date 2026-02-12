from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
A_SET = set(ALPHABET)


def _is_ascii_letter(ch: str) -> bool:
    o = ord(ch)
    return (65 <= o <= 90) or (97 <= o <= 122)


def _to_lower_ascii(ch: str) -> str:
    # Faster and more explicit than .lower() for single ASCII letters
    o = ord(ch)
    if 65 <= o <= 90:
        return chr(o + 32)
    return ch


def _apply_case(src: str, mapped_lower: str) -> str:
    # src is a single character; mapped_lower is lowercase a-z
    if "A" <= src <= "Z":
        return mapped_lower.upper()
    return mapped_lower


@dataclass(frozen=True)
class DecipherResult:
    """
    plaintext:
        The produced output with currently known mappings applied.
        Unknown letters are emitted unchanged by default (configurable).
    mapping:
        Known cipher->plain mapping (lowercase letters).
    reverse_mapping:
        Known plain->cipher mapping (lowercase letters).
    conflicts:
        List of detected inconsistencies when trying to learn mappings.
    """
    plaintext: str
    mapping: Dict[str, str]
    reverse_mapping: Dict[str, str]
    conflicts: List[str]


class SubstitutionDecipher:
    """
    Maintains and applies a *partial* one-to-one substitution mapping.

    The mapping is stored as cipher_letter -> plain_letter (both lowercase).
    Case is preserved at application time; non-letters are never changed.

    The decipher is robust:
      - validates inputs
      - enforces one-to-one constraints
      - detects conflicts without corrupting internal state
    """

    __slots__ = ("_c2p", "_p2c")

    def __init__(self, initial_mapping: Optional[Dict[str, str]] = None) -> None:
        self._c2p: Dict[str, str] = {}
        self._p2c: Dict[str, str] = {}
        if initial_mapping:
            # Load with full constraint checking
            conflicts = self.learn_pairs(initial_mapping.items())
            if conflicts:
                raise ValueError("Invalid initial_mapping:\n- " + "\n- ".join(conflicts))

    @property
    def mapping(self) -> Dict[str, str]:
        return dict(self._c2p)

    @property
    def reverse_mapping(self) -> Dict[str, str]:
        return dict(self._p2c)

    def decipher(
        self,
        message: str,
        *,
        unknown_policy: str = "keep",
        unknown_placeholder: str = "•",
    ) -> str:
        """
        unknown_policy:
            - "keep": keep unknown letters as-is (default)
            - "placeholder": replace unknown letters with unknown_placeholder
            - "lower_placeholder": placeholder but keeps the original case
        """
        if not isinstance(message, str):
            raise TypeError("message must be a str")

        if unknown_policy not in {"keep", "placeholder", "lower_placeholder"}:
            raise ValueError("unknown_policy must be 'keep', 'placeholder', or 'lower_placeholder'")

        out_chars: List[str] = []
        for ch in message:
            if not _is_ascii_letter(ch):
                out_chars.append(ch)
                continue

            c = _to_lower_ascii(ch)
            mapped = self._c2p.get(c)
            if mapped is not None:
                out_chars.append(_apply_case(ch, mapped))
            else:
                if unknown_policy == "keep":
                    out_chars.append(ch)
                elif unknown_policy == "placeholder":
                    out_chars.append(unknown_placeholder)
                else:  # "lower_placeholder"
                    # preserve case visually using placeholder (upper/lower same char)
                    out_chars.append(unknown_placeholder)
        return "".join(out_chars)

    def learn_pair(self, cipher_letter: str, plain_letter: str) -> Optional[str]:
        """
        Attempt to learn one mapping: cipher_letter -> plain_letter.

        Returns:
            None if successful (or already consistent),
            otherwise a conflict description string.
        """
        if not (isinstance(cipher_letter, str) and isinstance(plain_letter, str)):
            return "Both cipher_letter and plain_letter must be strings."

        if len(cipher_letter) != 1 or len(plain_letter) != 1:
            return "Both cipher_letter and plain_letter must be single characters."

        if not _is_ascii_letter(cipher_letter) or not _is_ascii_letter(plain_letter):
            return "Both cipher_letter and plain_letter must be ASCII letters A-Z or a-z."

        c = _to_lower_ascii(cipher_letter)
        p = _to_lower_ascii(plain_letter)

        # Enforce one-to-one constraints without mutating on conflict.
        existing_p = self._c2p.get(c)
        if existing_p is not None and existing_p != p:
            return f"Conflict: cipher '{c}' is already mapped to '{existing_p}', cannot map to '{p}'."

        existing_c = self._p2c.get(p)
        if existing_c is not None and existing_c != c:
            return f"Conflict: plain '{p}' is already mapped from cipher '{existing_c}', cannot also map from '{c}'."

        # Commit
        self._c2p[c] = p
        self._p2c[p] = c
        return None

    def learn_pairs(self, pairs: Iterable[Tuple[str, str]]) -> List[str]:
        """
        Attempt to learn multiple mappings. Continues through all pairs.
        Returns a list of conflicts encountered (empty if all succeeded).
        """
        conflicts: List[str] = []
        for c, p in pairs:
            err = self.learn_pair(c, p)
            if err is not None:
                conflicts.append(err)
        return conflicts

    def learn_from_crib(
        self,
        ciphertext: str,
        known_plaintext: str,
        *,
        strict_length: bool = True,
    ) -> List[str]:
        """
        Learn mappings from a 'crib' alignment: ciphertext aligned with known plaintext.
        Example: ciphertext="Xlmw", known_plaintext="This" would infer x->t, l->h, m->i, w->s

        Non-letters are ignored, but if strict_length=True, the two strings must be same length
        so positions align (recommended).
        """
        if not isinstance(ciphertext, str) or not isinstance(known_plaintext, str):
            raise TypeError("ciphertext and known_plaintext must be strings")

        if strict_length and len(ciphertext) != len(known_plaintext):
            return [f"Length mismatch: ciphertext has {len(ciphertext)} chars, plaintext has {len(known_plaintext)} chars."]

        conflicts: List[str] = []
        for i, (cc, pc) in enumerate(zip(ciphertext, known_plaintext)):
            if _is_ascii_letter(cc) and _is_ascii_letter(pc):
                err = self.learn_pair(cc, pc)
                if err is not None:
                    conflicts.append(f"At position {i}: {err}")
            else:
                # keep alignment logic simple: only learn from letter-letter pairs
                # (non-letters are not substituted and thus not informative)
                pass
        return conflicts

    def validate_complete_permutation(self) -> Optional[str]:
        """
        If mapping is complete (26 letters), verifies it is a permutation over a-z.
        Returns None if valid; otherwise a description of the problem.
        """
        if len(self._c2p) != 26:
            return f"Mapping is incomplete: {len(self._c2p)} / 26 letters known."

        keys = set(self._c2p.keys())
        vals = set(self._c2p.values())
        if keys != A_SET:
            missing = sorted(A_SET - keys)
            extra = sorted(keys - A_SET)
            return f"Cipher alphabet mismatch. Missing: {missing}, extra: {extra}"

        if vals != A_SET:
            missing = sorted(A_SET - vals)
            extra = sorted(vals - A_SET)
            return f"Plain alphabet mismatch. Missing: {missing}, extra: {extra}"

        # Reverse must be consistent too
        if len(self._p2c) != 26:
            return "Reverse mapping is inconsistent (duplicate plain letters detected)."

        return None


def decipher_and_update_dictionary(
    message: str,
    known_mapping: Optional[Dict[str, str]] = None,
    *,
    crib_ciphertext: Optional[str] = None,
    crib_plaintext: Optional[str] = None,
) -> DecipherResult:
    """
    Convenience wrapper:
      - creates a decipher with an initial mapping
      - optionally learns additional pairs from a crib
      - returns the deciphered output + updated dictionaries + conflicts

    If no crib is provided, this simply applies the known mapping to the message.
    """
    d = SubstitutionDecipher(initial_mapping=known_mapping or {})

    conflicts: List[str] = []
    if (crib_ciphertext is None) ^ (crib_plaintext is None):
        conflicts.append("Provide both crib_ciphertext and crib_plaintext, or neither.")
    elif crib_ciphertext is not None and crib_plaintext is not None:
        conflicts.extend(d.learn_from_crib(crib_ciphertext, crib_plaintext, strict_length=True))

    plaintext = d.decipher(message, unknown_policy="keep")
    return DecipherResult(
        plaintext=plaintext,
        mapping=d.mapping,
        reverse_mapping=d.reverse_mapping,
        conflicts=conflicts,
    )


# ----------------------------
# Example usage (safe to delete)
# ----------------------------
if __name__ == "__main__":
    # Suppose we know from context that "Xlmw" corresponds to "This"
    cipher = "Xlmw mw e xiwx. Xlmw mw srpc e xiwx!"
    crib_c = "Xlmw"
    crib_p = "This"

    result = decipher_and_update_dictionary(
        message=cipher,
        known_mapping={},          # start empty
        crib_ciphertext=crib_c,    # learn from crib
        crib_plaintext=crib_p,
    )

    print("Conflicts:", result.conflicts)
    print("Known mapping (cipher->plain):", result.mapping)
    print("Deciphered text:", result.plaintext)
