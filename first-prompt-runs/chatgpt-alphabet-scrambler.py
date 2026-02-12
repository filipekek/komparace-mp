import random
import string
from typing import Dict, Optional


def generate_cipher_mapping(seed: Optional[int] = None) -> Dict[str, str]:
    """
    Generate a substitution cipher mapping for the English alphabet.

    Each letter is mapped to exactly one other letter (a permutation).
    The mapping is bijective.

    Parameters:
        seed (Optional[int]): Seed for reproducible results.

    Returns:
        Dict[str, str]: Mapping from uppercase letters to uppercase letters.
    """
    alphabet = list(string.ascii_uppercase)
    scrambled = alphabet.copy()

    if seed is not None:
        random.seed(seed)

    random.shuffle(scrambled)

    return dict(zip(alphabet, scrambled))


def cipher_message(message: str, mapping: Dict[str, str]) -> str:
    """
    Cipher a message using the given substitution mapping.

    - Uppercase and lowercase letters are preserved.
    - Non-alphabetic characters remain unchanged.

    Parameters:
        message (str): Input message to cipher.
        mapping (Dict[str, str]): Cipher mapping (uppercase → uppercase).

    Returns:
        str: Ciphered message.
    """
    if not message:
        return ""

    result = []

    for char in message:
        if char.isalpha():
            is_upper = char.isupper()
            mapped_char = mapping[char.upper()]
            result.append(mapped_char if is_upper else mapped_char.lower())
        else:
            result.append(char)

    return "".join(result)


def main() -> None:
    """
    Run the alphabet scrambler interactively.
    """
    print("Alphabet Scrambler (Substitution Cipher)")
    print("---------------------------------------")

    message = input("Enter a message to cipher: ")

    if not message:
        print("Empty input. Nothing to cipher.")
        return

    mapping = generate_cipher_mapping()
    ciphered = cipher_message(message, mapping)

    print("\nCipher mapping:")
    for k in sorted(mapping):
        print(f"{k} → {mapping[k]}")

    print("\nCiphered message:")
    print(ciphered)


if __name__ == "__main__":
    main()
