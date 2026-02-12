"""
Substitution Cipher Decoder

This module implements a decoder for substitution ciphers where each letter
of the English alphabet is mapped to exactly one other letter (one-to-one mapping).
The decoder uses pattern matching and frequency analysis to deduce the cipher mapping.
"""

from collections import Counter, defaultdict
from typing import Dict, Set, List, Tuple, Optional
import re


class SubstitutionCipherDecoder:
    """
    A decoder for substitution ciphers using pattern matching and frequency analysis.
    """
    
    # English letter frequency (approximate percentages)
    ENGLISH_FREQ = {
        'E': 12.70, 'T': 9.06, 'A': 8.17, 'O': 7.51, 'I': 6.97,
        'N': 6.75, 'S': 6.33, 'H': 6.09, 'R': 5.99, 'D': 4.25,
        'L': 4.03, 'C': 2.78, 'U': 2.76, 'M': 2.41, 'W': 2.36,
        'F': 2.23, 'G': 2.02, 'Y': 1.97, 'P': 1.93, 'B': 1.29,
        'V': 0.98, 'K': 0.77, 'J': 0.15, 'X': 0.15, 'Q': 0.10, 'Z': 0.07
    }
    
    # Common English words for pattern matching
    COMMON_WORDS = {
        'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER',
        'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW',
        'ITS', 'MAY', 'OLD', 'SEE', 'SHE', 'TWO', 'WAY', 'WHO', 'BOY', 'DID',
        'LET', 'PUT', 'SAY', 'TOO', 'USE', 'THAT', 'WITH', 'HAVE', 'THIS',
        'FROM', 'THEY', 'BEEN', 'WERE', 'WHAT', 'YOUR', 'WILL', 'WHEN', 'THAN'
    }
    
    def __init__(self):
        """Initialize the decoder with an empty mapping."""
        self.cipher_to_plain: Dict[str, str] = {}
        self.plain_to_cipher: Dict[str, str] = {}
    
    def reset(self):
        """Reset the mapping to empty state."""
        self.cipher_to_plain.clear()
        self.plain_to_cipher.clear()
    
    def _extract_words(self, text: str) -> List[str]:
        """Extract alphabetic words from text (uppercase)."""
        words = re.findall(r'[A-Za-z]+', text)
        return [word.upper() for word in words if word]
    
    def _get_word_pattern(self, word: str) -> Tuple[int, ...]:
        """
        Get the pattern of a word based on letter repetitions.
        Example: 'HELLO' -> (0, 1, 2, 2, 3) - H=0, E=1, L=2, L=2, O=3
        """
        char_to_index = {}
        pattern = []
        next_index = 0
        
        for char in word.upper():
            if char not in char_to_index:
                char_to_index[char] = next_index
                next_index += 1
            pattern.append(char_to_index[char])
        
        return tuple(pattern)
    
    def _is_mapping_consistent(self, cipher_word: str, plain_word: str) -> bool:
        """
        Check if mapping cipher_word to plain_word is consistent with current mappings.
        """
        if len(cipher_word) != len(plain_word):
            return False
        
        cipher_word = cipher_word.upper()
        plain_word = plain_word.upper()
        
        # Check if patterns match
        if self._get_word_pattern(cipher_word) != self._get_word_pattern(plain_word):
            return False
        
        # Track proposed new mappings to check for conflicts within the word itself
        temp_cipher_to_plain = {}
        temp_plain_to_cipher = {}
        
        # Check consistency with existing mappings and within this word
        for c_char, p_char in zip(cipher_word, plain_word):
            # Check cipher -> plain mapping with existing dictionary
            if c_char in self.cipher_to_plain:
                if self.cipher_to_plain[c_char] != p_char:
                    return False
            
            # Check plain -> cipher mapping with existing dictionary (bijection)
            if p_char in self.plain_to_cipher:
                if self.plain_to_cipher[p_char] != c_char:
                    return False
            
            # Check for conflicts within the proposed word mapping
            if c_char in temp_cipher_to_plain:
                if temp_cipher_to_plain[c_char] != p_char:
                    return False
            else:
                temp_cipher_to_plain[c_char] = p_char
            
            if p_char in temp_plain_to_cipher:
                if temp_plain_to_cipher[p_char] != c_char:
                    return False
            else:
                temp_plain_to_cipher[p_char] = c_char
        
        return True
    
    def _apply_mapping(self, cipher_word: str, plain_word: str) -> bool:
        """
        Apply the mapping from cipher_word to plain_word.
        Returns True if successful, False if inconsistent.
        """
        if not self._is_mapping_consistent(cipher_word, plain_word):
            return False
        
        cipher_word = cipher_word.upper()
        plain_word = plain_word.upper()
        
        for c_char, p_char in zip(cipher_word, plain_word):
            self.cipher_to_plain[c_char] = p_char
            self.plain_to_cipher[p_char] = c_char
        
        return True
    
    def _frequency_analysis(self, text: str) -> Dict[str, float]:
        """Calculate letter frequency in the ciphertext."""
        letters = [c.upper() for c in text if c.isalpha()]
        if not letters:
            return {}
        
        counter = Counter(letters)
        total = sum(counter.values())
        
        return {char: (count / total) * 100 for char, count in counter.items()}
    
    def _match_patterns(self, cipher_words: List[str]) -> int:
        """
        Match cipher words with common English words based on patterns.
        Returns the number of new mappings found.
        """
        initial_mappings = len(self.cipher_to_plain)
        
        # Group words by pattern
        pattern_to_cipher = defaultdict(list)
        pattern_to_plain = defaultdict(list)
        
        for cipher_word in cipher_words:
            if len(cipher_word) >= 2:  # Focus on meaningful words
                pattern = self._get_word_pattern(cipher_word)
                pattern_to_cipher[pattern].append(cipher_word)
        
        for plain_word in self.COMMON_WORDS:
            pattern = self._get_word_pattern(plain_word)
            pattern_to_plain[pattern].append(plain_word)
        
        # Try to match patterns
        for pattern, cipher_word_list in pattern_to_cipher.items():
            if pattern in pattern_to_plain:
                plain_word_list = pattern_to_plain[pattern]
                
                # Try each combination
                for cipher_word in cipher_word_list:
                    for plain_word in plain_word_list:
                        if self._apply_mapping(cipher_word, plain_word):
                            break  # Move to next cipher word if mapping successful
        
        return len(self.cipher_to_plain) - initial_mappings
    
    def _frequency_based_guess(self, text: str) -> int:
        """
        Make educated guesses based on frequency analysis.
        Returns the number of new mappings found.
        """
        initial_mappings = len(self.cipher_to_plain)
        
        cipher_freq = self._frequency_analysis(text)
        if not cipher_freq:
            return 0
        
        # Get unmapped letters
        unmapped_cipher = [c for c in cipher_freq.keys() 
                          if c not in self.cipher_to_plain]
        unmapped_plain = [c for c in self.ENGLISH_FREQ.keys() 
                         if c not in self.plain_to_cipher]
        
        if not unmapped_cipher or not unmapped_plain:
            return 0
        
        # Sort by frequency
        cipher_sorted = sorted(unmapped_cipher, 
                              key=lambda x: cipher_freq[x], reverse=True)
        plain_sorted = sorted(unmapped_plain, 
                             key=lambda x: self.ENGLISH_FREQ[x], reverse=True)
        
        # Try to map most frequent to most frequent (cautiously)
        # Only map if frequencies are reasonably close
        for i in range(min(3, len(cipher_sorted), len(plain_sorted))):
            cipher_char = cipher_sorted[i]
            plain_char = plain_sorted[i]
            
            # This is a guess, so we apply it tentatively
            if self._is_mapping_consistent(cipher_char, plain_char):
                self._apply_mapping(cipher_char, plain_char)
        
        return len(self.cipher_to_plain) - initial_mappings
    
    def decode(self, ciphertext: str, 
               use_frequency: bool = True,
               use_patterns: bool = True) -> Tuple[str, Dict[str, str]]:
        """
        Decode the ciphertext and return the plaintext and mapping dictionary.
        
        Args:
            ciphertext: The encrypted message
            use_frequency: Whether to use frequency analysis
            use_patterns: Whether to use pattern matching
        
        Returns:
            A tuple of (decoded_text, mapping_dictionary)
        """
        self.reset()
        
        if not ciphertext:
            return "", {}
        
        # Extract words from ciphertext
        cipher_words = self._extract_words(ciphertext)
        
        # Apply pattern matching
        if use_patterns and cipher_words:
            self._match_patterns(cipher_words)
        
        # Apply frequency analysis
        if use_frequency:
            self._frequency_based_guess(ciphertext)
        
        # Decode the text
        decoded_text = self._decode_text(ciphertext)
        
        return decoded_text, dict(self.cipher_to_plain)
    
    def _decode_text(self, ciphertext: str) -> str:
        """
        Decode the ciphertext using the current mapping.
        Unknown characters are left as '?' or kept as original.
        """
        result = []
        
        for char in ciphertext:
            if char.isalpha():
                upper_char = char.upper()
                if upper_char in self.cipher_to_plain:
                    decoded_char = self.cipher_to_plain[upper_char]
                    # Preserve case
                    if char.islower():
                        result.append(decoded_char.lower())
                    else:
                        result.append(decoded_char)
                else:
                    result.append('_')  # Unknown character
            else:
                result.append(char)  # Non-alphabetic characters unchanged
        
        return ''.join(result)
    
    def add_known_mapping(self, cipher_letter: str, plain_letter: str) -> bool:
        """
        Manually add a known mapping.
        
        Args:
            cipher_letter: Letter in ciphertext
            plain_letter: Corresponding letter in plaintext
        
        Returns:
            True if mapping added successfully, False if inconsistent
        """
        if not cipher_letter.isalpha() or not plain_letter.isalpha():
            return False
        
        return self._apply_mapping(cipher_letter.upper(), plain_letter.upper())
    
    def get_mapping(self) -> Dict[str, str]:
        """Return the current cipher-to-plain mapping dictionary."""
        return dict(self.cipher_to_plain)


def main():
    """Example usage and interactive mode."""
    decoder = SubstitutionCipherDecoder()
    
    print("=" * 60)
    print("Substitution Cipher Decoder")
    print("=" * 60)
    
    # Example 1: Simple demonstration
    print("\nExample 1: Simple cipher")
    print("-" * 60)
    ciphertext1 = "URYYB JBEYQ"  # ROT13 of "HELLO WORLD"
    decoded1, mapping1 = decoder.decode(ciphertext1)
    print(f"Ciphertext: {ciphertext1}")
    print(f"Decoded:    {decoded1}")
    print(f"Mapping:    {mapping1}")
    
    # Example 2: Longer text
    print("\n\nExample 2: Longer text with common words")
    print("-" * 60)
    ciphertext2 = "QEB NRFZH YOLTK CLU GRJMP LSBO QEB IXWV ALD"
    # This is "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"
    decoded2, mapping2 = decoder.decode(ciphertext2)
    print(f"Ciphertext: {ciphertext2}")
    print(f"Decoded:    {decoded2}")
    print(f"Mapping:    {mapping2}")
    
    # Example 3: Interactive mode
    print("\n\nInteractive Mode")
    print("-" * 60)
    print("Enter a ciphertext to decode (or 'quit' to exit):")
    
    while True:
        user_input = input("\nCiphertext: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if not user_input:
            continue
        
        decoded, mapping = decoder.decode(user_input)
        print(f"Decoded:    {decoded}")
        print(f"Mapping found: {mapping}")
        print(f"Confidence: {len(mapping)}/26 letters mapped")


if __name__ == "__main__":
    main()