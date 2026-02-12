"""
Substitution Cipher Implementation

This module implements a substitution cipher that randomly permutes the alphabet
to create a cipher mapping. It preserves letter case and leaves non-alphabetic
characters unchanged.

Example:
    If alphabet ABCDE -> DCBEA, then:
    - Mapping: {A: D, B: C, C: B, D: E, E: A}
    - "BAD" encrypts to "CDE"
"""

import random
import string
from typing import Optional


class SubstitutionCipher:
    """A substitution cipher that uses a randomly permuted alphabet."""

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the cipher with a random alphabet permutation.

        Args:
            seed: Optional random seed for reproducible cipher generation.
        """
        self._cipher_map: dict[str, str] = {}
        self._decipher_map: dict[str, str] = {}
        self._generate_cipher(seed)

    def _generate_cipher(self, seed: Optional[int] = None) -> None:
        """
        Generate a random permutation of the alphabet for the cipher.

        Args:
            seed: Optional random seed for reproducibility.
        """
        alphabet = list(string.ascii_uppercase)

        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()

        shuffled = alphabet.copy()
        rng.shuffle(shuffled)

        # Create mapping for uppercase letters
        for original, substituted in zip(alphabet, shuffled):
            self._cipher_map[original] = substituted
            self._decipher_map[substituted] = original

    @property
    def cipher_map(self) -> dict[str, str]:
        """Return a copy of the cipher mapping (uppercase only)."""
        return self._cipher_map.copy()

    @property
    def decipher_map(self) -> dict[str, str]:
        """Return a copy of the decipher mapping (uppercase only)."""
        return self._decipher_map.copy()

    def _transform_char(self, char: str, mapping: dict[str, str]) -> str:
        """
        Transform a single character using the given mapping.

        Preserves case for letters and returns non-letters unchanged.

        Args:
            char: A single character to transform.
            mapping: The cipher or decipher mapping to use.

        Returns:
            The transformed character.
        """
        if not char:
            return char

        if char.isupper():
            return mapping.get(char, char)
        elif char.islower():
            upper_result = mapping.get(char.upper(), char.upper())
            return upper_result.lower()
        else:
            # Non-alphabetic characters pass through unchanged
            return char

    def encrypt(self, message: str) -> str:
        """
        Encrypt a message using the substitution cipher.

        Preserves letter case and leaves non-alphabetic characters unchanged.

        Args:
            message: The plaintext message to encrypt.

        Returns:
            The encrypted ciphertext.

        Examples:
            >>> cipher = SubstitutionCipher(seed=42)
            >>> cipher.encrypt("Hello, World!")
            'Iffmh, Chvmn!'
        """
        if not isinstance(message, str):
            raise TypeError(f"Message must be a string, got {type(message).__name__}")

        return ''.join(
            self._transform_char(char, self._cipher_map) for char in message
        )

    def decrypt(self, ciphertext: str) -> str:
        """
        Decrypt a message using the substitution cipher.

        Preserves letter case and leaves non-alphabetic characters unchanged.

        Args:
            ciphertext: The encrypted message to decrypt.

        Returns:
            The decrypted plaintext.

        Examples:
            >>> cipher = SubstitutionCipher(seed=42)
            >>> cipher.decrypt("Iffmh, Chvmn!")
            'Hello, World!'
        """
        if not isinstance(ciphertext, str):
            raise TypeError(f"Ciphertext must be a string, got {type(ciphertext).__name__}")

        return ''.join(
            self._transform_char(char, self._decipher_map) for char in ciphertext
        )

    def display_mapping(self) -> str:
        """
        Return a formatted string showing the cipher mapping.

        Returns:
            A string representation of the alphabet mapping.
        """
        original = string.ascii_uppercase
        substituted = ''.join(self._cipher_map[c] for c in original)
        return f"Original:    {original}\nSubstituted: {substituted}"


def main():
    """Interactive main function for the substitution cipher."""
    print("=" * 60)
    print("         SUBSTITUTION CIPHER")
    print("=" * 60)

    # Ask for optional seed
    seed_input = input("\nEnter a seed for reproducible cipher (or press Enter for random): ").strip()
    seed = int(seed_input) if seed_input.isdigit() else None

    cipher = SubstitutionCipher(seed=seed)

    print("\n" + "-" * 60)
    print("Generated Cipher Mapping:")
    print("-" * 60)
    print(cipher.display_mapping())
    print("-" * 60)

    while True:
        print("\nOptions:")
        print("  1. Encrypt a message")
        print("  2. Decrypt a message")
        print("  3. Show cipher mapping")
        print("  4. Generate new cipher")
        print("  5. Exit")

        choice = input("\nSelect an option (1-5): ").strip()

        if choice == "1":
            message = input("Enter message to encrypt: ")
            encrypted = cipher.encrypt(message)
            print(f"Encrypted: {encrypted}")

        elif choice == "2":
            message = input("Enter message to decrypt: ")
            decrypted = cipher.decrypt(message)
            print(f"Decrypted: {decrypted}")

        elif choice == "3":
            print("\n" + cipher.display_mapping())

        elif choice == "4":
            seed_input = input("Enter a seed (or press Enter for random): ").strip()
            seed = int(seed_input) if seed_input.isdigit() else None
            cipher = SubstitutionCipher(seed=seed)
            print("\nNew cipher generated!")
            print(cipher.display_mapping())

        elif choice == "5":
            print("Goodbye!")
            break

        else:
            print("Invalid option. Please select 1-5.")


if __name__ == "__main__":
    main()