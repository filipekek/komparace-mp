import random
import string

class SubstitutionCipher:
    def __init__(self):
        """
        Initialize the cipher. 
        The key is generated immediately upon initialization.
        """
        self.cipher_map = self._generate_key()

    def _generate_key(self) -> dict:
        """
        Generates a random substitution dictionary.
        
        Logic:
        1. Create a list of uppercase letters.
        2. Create a shuffled copy of that list.
        3. Map original -> shuffled.
        4. Mirror the mapping for lowercase to preserve case consistency.
        """
        original_upper = list(string.ascii_uppercase)
        shuffled_upper = original_upper[:] # Create a copy
        random.shuffle(shuffled_upper)
        
        # Create the dictionary for Uppercase
        # Example: {'A': 'Q', 'B': 'Z', ...}
        mapping = dict(zip(original_upper, shuffled_upper))
        
        # Extend the dictionary to Lowercase
        # If A maps to Q, then a must map to q to preserve case logic.
        for orig, sub in list(mapping.items()):
            mapping[orig.lower()] = sub.lower()
            
        return mapping

    def encrypt(self, text: str) -> str:
        """
        Encrypts the input text using the generated cipher map.
        
        Args:
            text (str): The string to encrypt.
            
        Returns:
            str: The encrypted string.
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string.")
        
        if not text:
            return ""

        # Using a list comprehension is faster and more pythonic 
        # than appending to a string in a loop.
        # .get(char, char) handles edge cases: if char is not in map 
        # (like numbers or punctuation), return the char itself.
        encrypted_chars = [self.cipher_map.get(char, char) for char in text]
        
        return "".join(encrypted_chars)

    def get_key_string(self) -> str:
        """Returns a readable representation of the Upper case key."""
        return ", ".join(f"{k}→{v}" for k, v in self.cipher_map.items() if k.isupper())

# --- Usage Example ---
if __name__ == "__main__":
    try:
        # 1. Initialize the Cipher
        cipher = SubstitutionCipher()
        
        # Optional: Print a snippet of the key to verify randomness
        print("Generated Key (Snippet):", list(cipher.cipher_map.items())[:5], "...")

        # 2. Get User Input
        user_message = input("Enter a message to cipher: ")
        
        # 3. Encrypt
        encrypted_message = cipher.encrypt(user_message)
        
        # 4. Output Results
        print("\n--- Results ---")
        print(f"Original:  {user_message}")
        print(f"Encrypted: {encrypted_message}")
        
    except Exception as e:
        print(f"An error occurred: {e}")