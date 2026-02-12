"""
Substitution Cipher Implementation
A robust implementation of a substitution cipher that randomly permutates
the alphabet to create an encryption key and cipher/decipher messages.
"""

import random
import string
from typing import Dict, Tuple


class SubstitutionCipher:
    """
    A substitution cipher that maps each letter to a randomly permutated letter.
    Preserves case and non-alphabetic characters.
    """
    
    def __init__(self, seed: int = None):
        """
        Initialize the substitution cipher with a random key.
        
        Args:
            seed: Optional random seed for reproducible key generation.
                  If None, uses system randomness.
        """
        self.original_alphabet = string.ascii_uppercase
        self.cipher_key = self._generate_cipher_key(seed)
        self.decipher_key = {v: k for k, v in self.cipher_key.items()}
    
    def _generate_cipher_key(self, seed: int = None) -> Dict[str, str]:
        """
        Generate a random substitution cipher key by permutating the alphabet.
        
        Args:
            seed: Optional random seed for reproducibility.
        
        Returns:
            Dictionary mapping each letter to its encrypted counterpart.
        """
        # Set seed if provided for reproducibility
        if seed is not None:
            random.seed(seed)
        
        # Create a shuffled copy of the alphabet
        shuffled_alphabet = list(self.original_alphabet)
        random.shuffle(shuffled_alphabet)
        
        # Create mapping dictionary
        cipher_key = {
            orig: shuffled 
            for orig, shuffled in zip(self.original_alphabet, shuffled_alphabet)
        }
        
        return cipher_key
    
    def get_key_display(self) -> str:
        """
        Get a formatted display of the cipher key.
        
        Returns:
            Formatted string showing the cipher key mapping.
        """
        lines = []
        lines.append("Cipher Key Mapping:")
        lines.append(f"Original:  {self.original_alphabet}")
        lines.append(f"Encrypted: {''.join(self.cipher_key[c] for c in self.original_alphabet)}")
        lines.append("\nDetailed Mapping:")
        
        # Display in rows of 13 for readability
        for i in range(0, 26, 13):
            mappings = [
                f"{orig}→{self.cipher_key[orig]}" 
                for orig in self.original_alphabet[i:i+13]
            ]
            lines.append("  ".join(mappings))
        
        return "\n".join(lines)
    
    def encrypt(self, message: str) -> str:
        """
        Encrypt a message using the substitution cipher.
        Preserves case and non-alphabetic characters.
        
        Args:
            message: The plaintext message to encrypt.
        
        Returns:
            The encrypted message.
        
        Raises:
            TypeError: If message is not a string.
        """
        if not isinstance(message, str):
            raise TypeError(f"Message must be a string, got {type(message).__name__}")
        
        if not message:  # Handle empty string
            return ""
        
        encrypted = []
        
        for char in message:
            if char.isupper():
                # Encrypt uppercase letter
                encrypted.append(self.cipher_key[char])
            elif char.islower():
                # Encrypt lowercase letter (convert to upper, encrypt, convert back)
                encrypted.append(self.cipher_key[char.upper()].lower())
            else:
                # Preserve non-alphabetic characters (spaces, punctuation, numbers, etc.)
                encrypted.append(char)
        
        return "".join(encrypted)
    
    def decrypt(self, encrypted_message: str) -> str:
        """
        Decrypt a message that was encrypted with this cipher.
        Preserves case and non-alphabetic characters.
        
        Args:
            encrypted_message: The encrypted message to decrypt.
        
        Returns:
            The decrypted (original) message.
        
        Raises:
            TypeError: If encrypted_message is not a string.
        """
        if not isinstance(encrypted_message, str):
            raise TypeError(f"Encrypted message must be a string, got {type(encrypted_message).__name__}")
        
        if not encrypted_message:  # Handle empty string
            return ""
        
        decrypted = []
        
        for char in encrypted_message:
            if char.isupper():
                # Decrypt uppercase letter
                decrypted.append(self.decipher_key[char])
            elif char.islower():
                # Decrypt lowercase letter
                decrypted.append(self.decipher_key[char.upper()].lower())
            else:
                # Preserve non-alphabetic characters
                decrypted.append(char)
        
        return "".join(decrypted)
    
    def get_cipher_dict(self) -> Dict[str, str]:
        """
        Get the cipher key dictionary.
        
        Returns:
            Copy of the cipher key mapping.
        """
        return self.cipher_key.copy()


def interactive_mode():
    """
    Run the cipher in interactive mode, allowing users to encrypt messages.
    """
    print("=" * 60)
    print("SUBSTITUTION CIPHER")
    print("=" * 60)
    
    # Ask if user wants to provide a seed for reproducibility
    use_seed = input("\nUse a seed for reproducible encryption? (y/n): ").strip().lower()
    
    if use_seed == 'y':
        try:
            seed = int(input("Enter seed (integer): "))
            cipher = SubstitutionCipher(seed=seed)
        except ValueError:
            print("Invalid seed. Using random seed.")
            cipher = SubstitutionCipher()
    else:
        cipher = SubstitutionCipher()
    
    # Display the cipher key
    print("\n" + cipher.get_key_display())
    print("\n" + "=" * 60)
    
    # Main loop
    while True:
        print("\nOptions:")
        print("1. Encrypt a message")
        print("2. Decrypt a message")
        print("3. Show cipher key")
        print("4. Generate new cipher key")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            message = input("\nEnter message to encrypt: ")
            encrypted = cipher.encrypt(message)
            print(f"Encrypted: {encrypted}")
            
        elif choice == '2':
            message = input("\nEnter message to decrypt: ")
            decrypted = cipher.decrypt(message)
            print(f"Decrypted: {decrypted}")
            
        elif choice == '3':
            print("\n" + cipher.get_key_display())
            
        elif choice == '4':
            use_seed = input("\nUse a seed for reproducible encryption? (y/n): ").strip().lower()
            if use_seed == 'y':
                try:
                    seed = int(input("Enter seed (integer): "))
                    cipher = SubstitutionCipher(seed=seed)
                except ValueError:
                    print("Invalid seed. Using random seed.")
                    cipher = SubstitutionCipher()
            else:
                cipher = SubstitutionCipher()
            print("\nNew cipher key generated!")
            print(cipher.get_key_display())
            
        elif choice == '5':
            print("\nGoodbye!")
            break
            
        else:
            print("\nInvalid choice. Please enter 1-5.")


def demonstrate_cipher():
    """
    Demonstrate the cipher with examples.
    """
    print("=" * 60)
    print("SUBSTITUTION CIPHER DEMONSTRATION")
    print("=" * 60)
    
    # Create cipher with a fixed seed for demonstration
    cipher = SubstitutionCipher(seed=42)
    
    print("\n" + cipher.get_key_display())
    
    # Test cases
    test_messages = [
        "BAD",
        "Hello, World!",
        "The Quick Brown Fox Jumps Over The Lazy Dog.",
        "12345 Test!@# with NUMBERS and symbols.",
        "",  # Empty string
        "ABC",
        "xyz"
    ]
    
    print("\n" + "=" * 60)
    print("TEST CASES:")
    print("=" * 60)
    
    for message in test_messages:
        encrypted = cipher.encrypt(message)
        decrypted = cipher.decrypt(encrypted)
        
        print(f"\nOriginal:  '{message}'")
        print(f"Encrypted: '{encrypted}'")
        print(f"Decrypted: '{decrypted}'")
        print(f"Match: {message == decrypted} ✓" if message == decrypted else f"Match: False ✗")


if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--demo":
            demonstrate_cipher()
        elif sys.argv[1] == "--help":
            print("Substitution Cipher")
            print("\nUsage:")
            print("  python substitution_cipher.py           - Run in interactive mode")
            print("  python substitution_cipher.py --demo    - Run demonstration")
            print("  python substitution_cipher.py --help    - Show this help message")
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Use --help for usage information")
    else:
        interactive_mode()