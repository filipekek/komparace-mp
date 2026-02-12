#!/usr/bin/env python3
"""
Random substitution cipher (monoalphabetic), preserving letter case.

- Generates a random permutation of A..Z and builds a mapping dict.
- Encrypts (and can decrypt) text while leaving non-letters unchanged.
- Robust: validates inputs and handles edge cases (empty strings, punctuation, etc.).
"""

from __future__ import annotations

import secrets
import string
from dataclasses import dataclass
from typing import Dict, Mapping, Optional


ALPHABET_UPPER = string.ascii_uppercase  # "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _random_permutation(alphabet: str = ALPHABET_UPPER) -> str:
    """
    Return a cryptographically strong random permutation of `alphabet`.
    """
    if not alphabet:
        raise ValueError("Alphabet must not be empty.")
    if len(set(alphabet)) != len(alphabet):
        raise ValueError("Alphabet must contain unique characters.")

    letters = list(alphabet)
    # Fisher–Yates shuffle using secrets for robust randomness
    for i in range(len(letters) - 1, 0, -1):
        j = secrets.randbelow(i + 1)  # 0 <= j <= i
        letters[i], letters[j] = letters[j], letters[i]
    return "".join(letters)


def _build_substitution_map(
    plain_alphabet: str = ALPHABET_UPPER,
    cipher_alphabet: Optional[str] = None,
) -> Dict[str, str]:
    """
    Build a mapping {plain_letter -> cipher_letter} for uppercase alphabets.
    If cipher_alphabet is None, it is generated randomly.
    """
    if not plain_alphabet:
        raise ValueError("plain_alphabet must not be empty.")
    if len(set(plain_alphabet)) != len(plain_alphabet):
        raise ValueError("plain_alphabet must contain unique characters.")

    if cipher_alphabet is None:
        cipher_alphabet = _random_permutation(plain_alphabet)
    else:
        if len(cipher_alphabet) != len(plain_alphabet):
            raise ValueError("cipher_alphabet must be the same length as plain_alphabet.")
        if len(set(cipher_alphabet)) != len(cipher_alphabet):
            raise ValueError("cipher_alphabet must contain unique characters.")
        # Ensure it's a permutation of the plain alphabet (same symbol set)
        if set(cipher_alphabet) != set(plain_alphabet):
            raise ValueError("cipher_alphabet must be a permutation of plain_alphabet.")

    return dict(zip(plain_alphabet, cipher_alphabet))


def _invert_map(mapping: Mapping[str, str]) -> Dict[str, str]:
    """
    Invert a one-to-one mapping {a->b} into {b->a}.
    """
    inverse: Dict[str, str] = {}
    for k, v in mapping.items():
        if v in inverse:
            raise ValueError("Mapping is not one-to-one; cannot invert safely.")
        inverse[v] = k
    return inverse


def _apply_substitution(text: str, mapping_upper: Mapping[str, str]) -> str:
    """
    Apply substitution to `text` preserving case.
    Only letters A-Z/a-z are substituted (based on mapping_upper).
    Other characters remain unchanged.
    """
    if text is None:
        raise ValueError("text must not be None.")

    out_chars: list[str] = []
    for ch in text:
        if "A" <= ch <= "Z":
            out_chars.append(mapping_upper.get(ch, ch))
        elif "a" <= ch <= "z":
            mapped = mapping_upper.get(ch.upper(), ch.upper())
            out_chars.append(mapped.lower())
        else:
            out_chars.append(ch)
    return "".join(out_chars)


@dataclass(frozen=True)
class SubstitutionCipher:
    """
    Substitution cipher that preserves case.
    `enc_map` is the uppercase mapping (A->X, B->Q, ...).
    """
    enc_map: Dict[str, str]

    @classmethod
    def random(cls) -> "SubstitutionCipher":
        return cls(enc_map=_build_substitution_map())

    @classmethod
    def from_cipher_alphabet(cls, cipher_alphabet: str) -> "SubstitutionCipher":
        # cipher_alphabet should be 26 unique letters, a permutation of A..Z
        return cls(enc_map=_build_substitution_map(ALPHABET_UPPER, cipher_alphabet.upper()))

    def encrypt(self, plaintext: str) -> str:
        return _apply_substitution(plaintext, self.enc_map)

    def decrypt(self, ciphertext: str) -> str:
        dec_map = _invert_map(self.enc_map)
        return _apply_substitution(ciphertext, dec_map)

    @property
    def cipher_alphabet(self) -> str:
        return "".join(self.enc_map[ch] for ch in ALPHABET_UPPER)


def main() -> None:
    cipher = SubstitutionCipher.random()

    print("Random substitution cipher generated.")
    print("Plain alphabet :", ALPHABET_UPPER)
    print("Cipher alphabet:", cipher.cipher_alphabet)
    print("Mapping (A->?):", cipher.enc_map)
    print()

    message = input("Enter a message to encrypt: ")
    encrypted = cipher.encrypt(message)
    print("\nEncrypted:", encrypted)

    # Optional: demonstrate decryption correctness
    decrypted = cipher.decrypt(encrypted)
    print("Decrypted:", decrypted)


if __name__ == "__main__":
    main()
