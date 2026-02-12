#!/usr/bin/env python3
"""
Substitution Cipher Decoder

This module implements a decoder for simple substitution ciphers where each letter
of the English alphabet is replaced by exactly one other letter (a permutation).

The decoder uses multiple strategies:
1. Frequency analysis - comparing letter frequencies to known English frequencies
2. Pattern matching - using word patterns to match against a dictionary
3. Interactive refinement - allowing manual corrections

Author: Claude
"""

import re
import string
from collections import Counter, defaultdict
from typing import Optional


# English letter frequencies (approximate, from large corpus analysis)
ENGLISH_FREQUENCIES = {
    'E': 12.70, 'T': 9.06, 'A': 8.17, 'O': 7.51, 'I': 6.97,
    'N': 6.75, 'S': 6.33, 'H': 6.09, 'R': 5.99, 'D': 4.25,
    'L': 4.03, 'C': 2.78, 'U': 2.76, 'M': 2.41, 'W': 2.36,
    'F': 2.23, 'G': 2.02, 'Y': 1.97, 'P': 1.93, 'B': 1.29,
    'V': 0.98, 'K': 0.77, 'J': 0.15, 'X': 0.15, 'Q': 0.10,
    'Z': 0.07
}

# Common English words for pattern matching
COMMON_WORDS = [
    # Single letters
    'a', 'i',
    # Two letters
    'to', 'of', 'in', 'it', 'is', 'be', 'as', 'at', 'so', 'we', 'he', 'by',
    'or', 'on', 'do', 'if', 'me', 'my', 'up', 'an', 'go', 'no', 'us', 'am',
    # Three letters
    'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
    'her', 'was', 'one', 'our', 'out', 'are', 'has', 'his', 'how', 'its',
    'let', 'may', 'new', 'now', 'old', 'see', 'way', 'who', 'boy', 'did',
    'get', 'him', 'got', 'say', 'she', 'too', 'use',
    # Four letters
    'that', 'with', 'have', 'this', 'will', 'your', 'from', 'they', 'been',
    'call', 'each', 'find', 'here', 'just', 'know', 'like', 'long', 'make',
    'more', 'only', 'over', 'such', 'take', 'than', 'them', 'then', 'time',
    'very', 'when', 'come', 'made', 'many', 'some', 'what', 'year', 'said',
    'also', 'back', 'been', 'come', 'could', 'good', 'into', 'look', 'most',
    'much', 'must', 'need', 'part', 'same', 'tell', 'well', 'went', 'were',
    'work', 'down', 'even', 'give', 'last', 'life', 'name', 'next', 'read',
    'right', 'still', 'think', 'want', 'where', 'which', 'world', 'would',
    # Five letters
    'about', 'after', 'again', 'being', 'could', 'every', 'first', 'found',
    'great', 'house', 'large', 'learn', 'never', 'other', 'place', 'plant',
    'point', 'right', 'small', 'sound', 'spell', 'still', 'study', 'their',
    'there', 'these', 'thing', 'think', 'three', 'water', 'where', 'which',
    'world', 'would', 'write', 'years', 'because', 'before', 'between',
    'should', 'through', 'people', 'little', 'different', 'important'
]


def get_word_pattern(word: str) -> str:
    """
    Generate a pattern for a word where each unique letter gets a unique number.
    
    Example: 'hello' -> '0.1.2.2.3', 'abba' -> '0.1.1.0'
    
    Args:
        word: The word to generate a pattern for (case insensitive)
        
    Returns:
        A string pattern representing the letter structure
    """
    word = word.upper()
    letter_to_num = {}
    pattern = []
    counter = 0
    
    for letter in word:
        if letter not in letter_to_num:
            letter_to_num[letter] = str(counter)
            counter += 1
        pattern.append(letter_to_num[letter])
    
    return '.'.join(pattern)


def build_pattern_dictionary(words: list[str]) -> dict[str, list[str]]:
    """
    Build a dictionary mapping word patterns to lists of words with that pattern.
    
    Args:
        words: List of words to process
        
    Returns:
        Dictionary mapping patterns to word lists
    """
    pattern_dict = defaultdict(list)
    for word in words:
        if word.isalpha():
            pattern = get_word_pattern(word)
            pattern_dict[pattern].append(word.upper())
    return dict(pattern_dict)


# Pre-build pattern dictionary from common words
PATTERN_DICT = build_pattern_dictionary(COMMON_WORDS)


def load_system_dictionary() -> dict[str, list[str]]:
    """
    Load words from system dictionary if available and build pattern dictionary.
    Falls back to COMMON_WORDS if system dictionary not found.
    
    Returns:
        Pattern dictionary mapping patterns to word lists
    """
    dict_paths = [
        '/usr/share/dict/words',
        '/usr/share/dict/american-english',
        '/usr/share/dict/british-english',
    ]
    
    words = set(COMMON_WORDS)
    
    for path in dict_paths:
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    word = line.strip().lower()
                    # Only include words that are purely alphabetic
                    if word.isalpha() and 2 <= len(word) <= 15:
                        words.add(word)
        except (FileNotFoundError, PermissionError):
            continue
    
    return build_pattern_dictionary(list(words))


# Try to load extended dictionary
try:
    EXTENDED_PATTERN_DICT = load_system_dictionary()
except Exception:
    EXTENDED_PATTERN_DICT = PATTERN_DICT


class SubstitutionCipherDecoder:
    """
    A class to decode substitution ciphers using frequency analysis and pattern matching.
    
    Attributes:
        cipher_to_plain: Dictionary mapping cipher letters to plaintext letters
        plain_to_cipher: Dictionary mapping plaintext letters to cipher letters
        ciphertext: The original encrypted message
    """
    
    def __init__(self):
        """Initialize the decoder with empty mappings."""
        self.cipher_to_plain: dict[str, str] = {}
        self.plain_to_cipher: dict[str, str] = {}
        self.ciphertext: str = ""
        self._candidates: dict[str, set[str]] = {
            letter: set(string.ascii_uppercase) for letter in string.ascii_uppercase
        }
    
    def reset(self) -> None:
        """Reset all mappings and candidates."""
        self.cipher_to_plain.clear()
        self.plain_to_cipher.clear()
        self.ciphertext = ""
        self._candidates = {
            letter: set(string.ascii_uppercase) for letter in string.ascii_uppercase
        }
    
    def set_mapping(self, cipher_letter: str, plain_letter: str) -> bool:
        """
        Set a mapping from a cipher letter to a plaintext letter.
        
        Args:
            cipher_letter: The letter in the ciphertext
            plain_letter: The corresponding plaintext letter
            
        Returns:
            True if mapping was set successfully, False if it conflicts
        """
        cipher_letter = cipher_letter.upper()
        plain_letter = plain_letter.upper()
        
        # Validate inputs
        if not (cipher_letter.isalpha() and len(cipher_letter) == 1):
            return False
        if not (plain_letter.isalpha() and len(plain_letter) == 1):
            return False
        
        # Check for conflicts
        if cipher_letter in self.cipher_to_plain:
            if self.cipher_to_plain[cipher_letter] != plain_letter:
                return False
        if plain_letter in self.plain_to_cipher:
            if self.plain_to_cipher[plain_letter] != cipher_letter:
                return False
        
        # Set the mapping
        self.cipher_to_plain[cipher_letter] = plain_letter
        self.plain_to_cipher[plain_letter] = cipher_letter
        
        # Update candidates
        self._candidates[cipher_letter] = {plain_letter}
        for c in string.ascii_uppercase:
            if c != cipher_letter:
                self._candidates[c].discard(plain_letter)
        
        return True
    
    def remove_mapping(self, cipher_letter: str) -> bool:
        """
        Remove a mapping for a cipher letter.
        
        Args:
            cipher_letter: The cipher letter to unmap
            
        Returns:
            True if mapping was removed, False if it didn't exist
        """
        cipher_letter = cipher_letter.upper()
        
        if cipher_letter not in self.cipher_to_plain:
            return False
        
        plain_letter = self.cipher_to_plain[cipher_letter]
        del self.cipher_to_plain[cipher_letter]
        del self.plain_to_cipher[plain_letter]
        
        # Reset candidates for this letter
        self._candidates[cipher_letter] = set(string.ascii_uppercase) - set(self.plain_to_cipher.keys())
        
        return True
    
    def analyze_frequencies(self, text: str) -> dict[str, float]:
        """
        Analyze letter frequencies in the given text.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary mapping letters to their percentage frequencies
        """
        letters_only = [c.upper() for c in text if c.isalpha()]
        total = len(letters_only)
        
        if total == 0:
            return {}
        
        counts = Counter(letters_only)
        return {letter: (count / total) * 100 for letter, count in counts.items()}
    
    def suggest_frequency_mapping(self, ciphertext: str) -> dict[str, str]:
        """
        Suggest a mapping based on frequency analysis.
        
        Args:
            ciphertext: The encrypted text
            
        Returns:
            Dictionary of suggested cipher->plain mappings
        """
        cipher_freqs = self.analyze_frequencies(ciphertext)
        
        if not cipher_freqs:
            return {}
        
        # Sort cipher letters by frequency (descending)
        sorted_cipher = sorted(cipher_freqs.keys(), key=lambda x: cipher_freqs[x], reverse=True)
        
        # Sort English letters by frequency (descending)
        sorted_english = sorted(ENGLISH_FREQUENCIES.keys(), key=lambda x: ENGLISH_FREQUENCIES[x], reverse=True)
        
        # Create initial mapping suggestion
        suggestions = {}
        for i, cipher_letter in enumerate(sorted_cipher):
            if i < len(sorted_english):
                suggestions[cipher_letter] = sorted_english[i]
        
        return suggestions
    
    def get_cipher_words(self, ciphertext: str) -> list[str]:
        """
        Extract words from ciphertext (alphabetic sequences only).
        
        Args:
            ciphertext: The encrypted text
            
        Returns:
            List of cipher words
        """
        return [word.upper() for word in re.findall(r'[a-zA-Z]+', ciphertext)]
    
    def find_pattern_matches(self, cipher_word: str, use_extended: bool = True) -> list[str]:
        """
        Find dictionary words that match the pattern of a cipher word.
        
        Args:
            cipher_word: The encrypted word
            use_extended: Whether to use extended system dictionary
            
        Returns:
            List of possible plaintext words
        """
        pattern = get_word_pattern(cipher_word.upper())
        dict_to_use = EXTENDED_PATTERN_DICT if use_extended else PATTERN_DICT
        return dict_to_use.get(pattern, [])
    
    def is_mapping_consistent(self, cipher_word: str, plain_word: str) -> bool:
        """
        Check if mapping a cipher word to a plain word is consistent with known mappings.
        
        Args:
            cipher_word: The encrypted word
            plain_word: The proposed plaintext word
            
        Returns:
            True if consistent, False otherwise
        """
        cipher_word = cipher_word.upper()
        plain_word = plain_word.upper()
        
        if len(cipher_word) != len(plain_word):
            return False
        
        for c, p in zip(cipher_word, plain_word):
            # Check if cipher letter already maps to a different plain letter
            if c in self.cipher_to_plain and self.cipher_to_plain[c] != p:
                return False
            # Check if plain letter already maps to a different cipher letter
            if p in self.plain_to_cipher and self.plain_to_cipher[p] != c:
                return False
        
        return True
    
    def apply_word_mapping(self, cipher_word: str, plain_word: str) -> bool:
        """
        Apply mappings from a cipher word to a plain word.
        
        Args:
            cipher_word: The encrypted word
            plain_word: The plaintext word
            
        Returns:
            True if all mappings were applied successfully
        """
        cipher_word = cipher_word.upper()
        plain_word = plain_word.upper()
        
        if not self.is_mapping_consistent(cipher_word, plain_word):
            return False
        
        for c, p in zip(cipher_word, plain_word):
            if c not in self.cipher_to_plain:
                self.set_mapping(c, p)
        
        return True
    
    def decode_text(self, ciphertext: str, use_known_only: bool = True) -> str:
        """
        Decode ciphertext using the current mappings.
        
        Args:
            ciphertext: The encrypted text
            use_known_only: If True, leave unknown letters as-is; if False, show as ?
            
        Returns:
            The decoded text (or partially decoded with unknowns)
        """
        result = []
        
        for char in ciphertext:
            if char.upper() in self.cipher_to_plain:
                decoded = self.cipher_to_plain[char.upper()]
                # Preserve case
                if char.islower():
                    decoded = decoded.lower()
                result.append(decoded)
            elif char.isalpha():
                result.append(char if use_known_only else '?')
            else:
                # Non-alphabetic characters remain unchanged
                result.append(char)
        
        return ''.join(result)
    
    def auto_decode(self, ciphertext: str, max_iterations: int = 100) -> str:
        """
        Attempt to automatically decode the ciphertext using pattern matching.
        
        This uses a constraint-based approach, matching cipher words to dictionary
        words and propagating constraints.
        
        Args:
            ciphertext: The encrypted text
            max_iterations: Maximum iterations for refinement
            
        Returns:
            The best decoded text found
        """
        self.ciphertext = ciphertext
        cipher_words = self.get_cipher_words(ciphertext)
        
        if not cipher_words:
            return ciphertext
        
        # Get unique words and their frequencies
        word_counts = Counter(cipher_words)
        unique_words = list(set(cipher_words))
        
        # Sort words by: (1) frequency (more frequent = more constraining if correct)
        # (2) length (longer = more unique patterns), (3) fewer matches (more constraining)
        word_matches = []
        for word in unique_words:
            matches = self.find_pattern_matches(word)
            if matches:
                # Prioritize common words and longer words with fewer matches
                score = word_counts[word] * len(word) / (len(matches) + 1)
                word_matches.append((word, matches, score))
        
        # Sort by score descending (most constraining first)
        word_matches.sort(key=lambda x: x[2], reverse=True)
        
        # Try to find consistent mappings using backtracking
        best_mapping = {}
        best_score = 0
        
        def try_mapping(word_idx: int, current_c2p: dict, current_p2c: dict) -> tuple[dict, int]:
            """Try to extend mapping with remaining words."""
            nonlocal best_mapping, best_score
            
            # Score current mapping
            score = len(current_c2p)
            if score > best_score:
                best_score = score
                best_mapping = dict(current_c2p)
            
            if word_idx >= len(word_matches):
                return current_c2p, score
            
            cipher_word, matches, _ = word_matches[word_idx]
            
            # Try each possible match
            found_any = False
            for plain_word in matches[:20]:  # Limit candidates for performance
                # Check consistency
                new_c2p = dict(current_c2p)
                new_p2c = dict(current_p2c)
                consistent = True
                
                for c, p in zip(cipher_word, plain_word):
                    if c in new_c2p:
                        if new_c2p[c] != p:
                            consistent = False
                            break
                    elif p in new_p2c:
                        if new_p2c[p] != c:
                            consistent = False
                            break
                    else:
                        new_c2p[c] = p
                        new_p2c[p] = c
                
                if consistent:
                    found_any = True
                    try_mapping(word_idx + 1, new_c2p, new_p2c)
            
            # If no matches worked, skip this word
            if not found_any:
                try_mapping(word_idx + 1, current_c2p, current_p2c)
            
            return best_mapping, best_score
        
        # Run backtracking search (with iteration limit)
        if word_matches:
            try_mapping(0, {}, {})
        
        # Apply best mapping found
        for cipher, plain in best_mapping.items():
            self.set_mapping(cipher, plain)
        
        # If we haven't mapped many letters, fall back to frequency analysis
        if len(self.cipher_to_plain) < 10:
            suggestions = self.suggest_frequency_mapping(ciphertext)
            for cipher, plain in suggestions.items():
                if cipher not in self.cipher_to_plain and plain not in self.plain_to_cipher:
                    self.set_mapping(cipher, plain)
        
        return self.decode_text(ciphertext)
    
    def get_mapping_dict(self) -> dict[str, str]:
        """
        Get the current cipher-to-plaintext mapping dictionary.
        
        Returns:
            Dictionary of known mappings (cipher -> plain)
        """
        return dict(self.cipher_to_plain)
    
    def get_reverse_mapping_dict(self) -> dict[str, str]:
        """
        Get the current plaintext-to-cipher mapping dictionary.
        
        Returns:
            Dictionary of known mappings (plain -> cipher)
        """
        return dict(self.plain_to_cipher)
    
    def display_status(self) -> str:
        """
        Generate a status display of current mappings.
        
        Returns:
            Formatted string showing the mapping status
        """
        lines = []
        lines.append("Current Cipher-to-Plain Mappings:")
        lines.append("-" * 40)
        
        if not self.cipher_to_plain:
            lines.append("(No mappings established yet)")
        else:
            # Display in alphabetical order
            sorted_mappings = sorted(self.cipher_to_plain.items())
            mapping_str = ", ".join(f"{c}→{p}" for c, p in sorted_mappings)
            lines.append(mapping_str)
        
        lines.append("")
        lines.append("Unmapped cipher letters: " + 
                    "".join(sorted(set(string.ascii_uppercase) - set(self.cipher_to_plain.keys()))))
        lines.append("Unmapped plain letters: " + 
                    "".join(sorted(set(string.ascii_uppercase) - set(self.plain_to_cipher.keys()))))
        
        return "\n".join(lines)


def interactive_decoder():
    """
    Run an interactive session for decoding substitution ciphers.
    """
    decoder = SubstitutionCipherDecoder()
    
    print("=" * 60)
    print("Substitution Cipher Decoder")
    print("=" * 60)
    print("\nCommands:")
    print("  enter/e <text>  - Enter ciphertext to decode")
    print("  auto/a          - Attempt automatic decoding")
    print("  map/m <C> <P>   - Set mapping: cipher C -> plain P")
    print("  unmap/u <C>     - Remove mapping for cipher letter C")
    print("  show/s          - Show current decoded text")
    print("  status/t        - Show mapping status")
    print("  freq/f          - Show frequency analysis")
    print("  suggest/g       - Suggest mappings based on frequency")
    print("  words/w         - Show cipher words and possible matches")
    print("  reset/r         - Reset all mappings")
    print("  dict/d          - Show mapping dictionary")
    print("  help/h          - Show this help")
    print("  quit/q          - Exit")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\n> ").strip()
        except EOFError:
            break
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        
        if not user_input:
            continue
        
        parts = user_input.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if command in ('quit', 'q', 'exit'):
            print("Goodbye!")
            break
        
        elif command in ('help', 'h'):
            print("\nCommands:")
            print("  enter/e <text>  - Enter ciphertext to decode")
            print("  auto/a          - Attempt automatic decoding")
            print("  map/m <C> <P>   - Set mapping: cipher C -> plain P")
            print("  unmap/u <C>     - Remove mapping for cipher letter C")
            print("  show/s          - Show current decoded text")
            print("  status/t        - Show mapping status")
            print("  freq/f          - Show frequency analysis")
            print("  suggest/g       - Suggest mappings based on frequency")
            print("  words/w         - Show cipher words and possible matches")
            print("  reset/r         - Reset all mappings")
            print("  dict/d          - Show mapping dictionary")
            print("  quit/q          - Exit")
        
        elif command in ('enter', 'e'):
            if not args:
                print("Please provide ciphertext: enter <text>")
                continue
            decoder.ciphertext = args
            print(f"Ciphertext set: {args}")
            print(f"Length: {len(args)} characters, {len([c for c in args if c.isalpha()])} letters")
        
        elif command in ('auto', 'a'):
            if not decoder.ciphertext:
                print("No ciphertext entered. Use 'enter <text>' first.")
                continue
            decoded = decoder.auto_decode(decoder.ciphertext)
            print(f"\nOriginal:  {decoder.ciphertext}")
            print(f"Decoded:   {decoded}")
            print(f"\nMappings found: {len(decoder.cipher_to_plain)}/26")
        
        elif command in ('map', 'm'):
            map_parts = args.split()
            if len(map_parts) != 2:
                print("Usage: map <cipher_letter> <plain_letter>")
                continue
            cipher_letter, plain_letter = map_parts
            if decoder.set_mapping(cipher_letter, plain_letter):
                print(f"Mapped: {cipher_letter.upper()} → {plain_letter.upper()}")
                if decoder.ciphertext:
                    print(f"Current decode: {decoder.decode_text(decoder.ciphertext)}")
            else:
                print("Failed to set mapping (conflict with existing mapping)")
        
        elif command in ('unmap', 'u'):
            if not args or not args[0].isalpha():
                print("Usage: unmap <cipher_letter>")
                continue
            if decoder.remove_mapping(args[0]):
                print(f"Removed mapping for: {args[0].upper()}")
            else:
                print(f"No mapping exists for: {args[0].upper()}")
        
        elif command in ('show', 's'):
            if not decoder.ciphertext:
                print("No ciphertext entered. Use 'enter <text>' first.")
                continue
            print(f"Original:  {decoder.ciphertext}")
            print(f"Decoded:   {decoder.decode_text(decoder.ciphertext)}")
        
        elif command in ('status', 't'):
            print(decoder.display_status())
        
        elif command in ('freq', 'f'):
            if not decoder.ciphertext:
                print("No ciphertext entered. Use 'enter <text>' first.")
                continue
            freqs = decoder.analyze_frequencies(decoder.ciphertext)
            sorted_freqs = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
            print("\nCiphertext letter frequencies:")
            for letter, freq in sorted_freqs:
                bar = '█' * int(freq)
                print(f"  {letter}: {freq:5.2f}% {bar}")
        
        elif command in ('suggest', 'g'):
            if not decoder.ciphertext:
                print("No ciphertext entered. Use 'enter <text>' first.")
                continue
            suggestions = decoder.suggest_frequency_mapping(decoder.ciphertext)
            print("\nSuggested mappings based on frequency analysis:")
            cipher_freqs = decoder.analyze_frequencies(decoder.ciphertext)
            for cipher, plain in sorted(suggestions.items(), 
                                       key=lambda x: cipher_freqs.get(x[0], 0), 
                                       reverse=True):
                status = ""
                if cipher in decoder.cipher_to_plain:
                    actual = decoder.cipher_to_plain[cipher]
                    status = f" (currently: {actual})"
                print(f"  {cipher} → {plain}{status}")
        
        elif command in ('words', 'w'):
            if not decoder.ciphertext:
                print("No ciphertext entered. Use 'enter <text>' first.")
                continue
            words = set(decoder.get_cipher_words(decoder.ciphertext))
            print(f"\nCipher words ({len(words)} unique):")
            for word in sorted(words, key=len):
                matches = decoder.find_pattern_matches(word)
                matches_str = ", ".join(matches[:5])
                if len(matches) > 5:
                    matches_str += f"... ({len(matches)} total)"
                elif not matches:
                    matches_str = "(no matches in dictionary)"
                print(f"  {word} ({get_word_pattern(word)}): {matches_str}")
        
        elif command in ('reset', 'r'):
            decoder.reset()
            print("All mappings reset.")
        
        elif command in ('dict', 'd'):
            mapping = decoder.get_mapping_dict()
            if not mapping:
                print("No mappings established yet.")
            else:
                print("\nMapping Dictionary (cipher → plain):")
                print(mapping)
                print("\nReverse Dictionary (plain → cipher):")
                print(decoder.get_reverse_mapping_dict())
        
        else:
            print(f"Unknown command: {command}. Type 'help' for available commands.")


def decode_message(ciphertext: str) -> tuple[str, dict[str, str]]:
    """
    Convenience function to decode a message and return results.
    
    Args:
        ciphertext: The encrypted message
        
    Returns:
        Tuple of (decoded_text, mapping_dictionary)
    """
    decoder = SubstitutionCipherDecoder()
    decoded = decoder.auto_decode(ciphertext)
    return decoded, decoder.get_mapping_dict()


# Example usage and demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("Substitution Cipher Decoder - Demo")
    print("=" * 60)
    
    # Demo with a simple example
    print("\n--- Simple Example ---")
    # Using the mapping from the prompt: A->D, B->C, C->B, D->E, E->A
    # "BAD" encrypts to "CDE"
    example_cipher = "CDE"
    decoder = SubstitutionCipherDecoder()
    
    # Manually set the known mappings for this small example
    decoder.set_mapping('C', 'B')
    decoder.set_mapping('D', 'A')
    decoder.set_mapping('E', 'D')
    
    decoded = decoder.decode_text(example_cipher)
    print(f"Ciphertext: {example_cipher}")
    print(f"Decoded: {decoded}")
    print(f"Mapping: {decoder.get_mapping_dict()}")
    
    # Demo with a longer example that can be auto-decoded
    print("\n--- Auto-Decode Example ---")
    # This is "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG" encrypted with a substitution cipher
    # Using cipher: ZYXWVUTSRQPONMLKJIHGFEDCBA (reverse alphabet)
    test_cipher = "GSV JFRXP YILDM ULC QFNKH LEVI GSV OZAB WLT"
    
    decoder2 = SubstitutionCipherDecoder()
    decoded2 = decoder2.auto_decode(test_cipher)
    
    print(f"Ciphertext: {test_cipher}")
    print(f"Decoded: {decoded2}")
    print(f"\nMappings found ({len(decoder2.cipher_to_plain)}/26):")
    print(decoder2.get_mapping_dict())
    
    # Run interactive mode
    print("\n" + "=" * 60)
    print("Starting Interactive Mode...")
    print("=" * 60)
    interactive_decoder()