import random
import string
import sys
from collections import Counter

class SubstitutionCipherSolver:
    """
    Deciphers a substitution cipher without a key using Hill-Climbing optimization.
    Features:
    - Initial guess based on letter frequency.
    - Optimization loop using English Bigram scoring.
    - Preserves case and ignores non-alphabetic characters.
    """

    def __init__(self):
        self.alphabet = list(string.ascii_uppercase)
        
        # Standard English Letter Frequencies (Probability)
        self.english_freq = {
            'E': 12.70, 'T': 9.06, 'A': 8.17, 'O': 7.51, 'I': 6.97, 'N': 6.75, 'S': 6.33,
            'H': 6.09, 'R': 5.99, 'D': 4.25, 'L': 4.03, 'C': 2.78, 'U': 2.76, 'M': 2.41,
            'W': 2.36, 'F': 2.23, 'G': 2.02, 'Y': 1.97, 'P': 1.93, 'B': 1.29, 'V': 0.98,
            'K': 0.77, 'J': 0.15, 'X': 0.15, 'Q': 0.10, 'Z': 0.07
        }

        # Top 50 English Bigrams with approximate log-probability scores.
        # This acts as our "English Dictionary" to score how readable the text is.
        self.bigrams = {
            'TH': 10, 'HE': 9,  'IN': 9,  'ER': 8,  'AN': 8,  'RE': 7,
            'ON': 7,  'AT': 7,  'EN': 7,  'ND': 6,  'TI': 6,  'ES': 6,
            'OR': 6,  'TE': 6,  'OF': 6,  'ED': 5,  'IS': 5,  'IT': 5,
            'AL': 5,  'AR': 5,  'ST': 5,  'TO': 5,  'NT': 4,  'NG': 4,
            'SE': 4,  'HA': 4,  'AS': 4,  'OU': 4,  'IO': 4,  'LE': 4,
            'VE': 3,  'CO': 3,  'ME': 3,  'DE': 3,  'HI': 3,  'RI': 3,
            'RO': 3,  'IC': 3,  'NE': 3,  'EA': 3,  'RA': 3,  'CE': 3
        }

    def _get_cipher_freq(self, text):
        """Calculates frequency of letters in the ciphertext."""
        filtered = [c for c in text if c in self.alphabet]
        if not filtered: return {}
        counts = Counter(filtered)
        total = sum(counts.values())
        return {k: (v/total)*100 for k, v in counts.items()}

    def _generate_initial_key(self, ciphertext):
        """
        Generates a starting key by matching sorted frequencies of 
        ciphertext directly to sorted frequencies of English.
        """
        cipher_f = self._get_cipher_freq(ciphertext)
        
        # Sort both lists by frequency (High -> Low)
        sorted_cipher = sorted(cipher_f.keys(), key=lambda x: cipher_f[x], reverse=True)
        sorted_english = sorted(self.english_freq.keys(), key=lambda x: self.english_freq[x], reverse=True)
        
        # Create initial mapping
        mapping = {}
        used_plain = set()
        
        # Map frequent letters first
        for c, e in zip(sorted_cipher, sorted_english):
            mapping[c] = e
            used_plain.add(e)
            
        # Fill remaining letters randomly
        unused_cipher = [c for c in self.alphabet if c not in mapping]
        unused_plain = [p for p in self.alphabet if p not in used_plain]
        
        for c, p in zip(unused_cipher, unused_plain):
            mapping[c] = p
            
        # Convert to list format [Plain for A, Plain for B, ...] for fast swapping
        return [mapping[c] for c in self.alphabet]

    def _decrypt(self, text, key_list):
        """Fast decryption using translation table."""
        # Key list is the target values for A-Z
        trans_table = str.maketrans("".join(self.alphabet), "".join(key_list))
        return text.translate(trans_table)

    def _score(self, text):
        """Scores text based on Bigram frequency (Higher is better)."""
        score = 0
        for i in range(len(text) - 1):
            pair = text[i:i+2]
            score += self.bigrams.get(pair, -1) # Penalty for non-English bigrams
        return score

    def solve(self, ciphertext, steps=3000):
        """
        Main function to solve the cipher.
        Returns: (decrypted_text, key_dictionary)
        """
        if not ciphertext: return "", {}
        
        # Work with uppercase for analysis to simplify logic
        msg_upper = ciphertext.upper()
        
        # 1. Initial Guess
        best_key = self._generate_initial_key(msg_upper)
        current_text = self._decrypt(msg_upper, best_key)
        best_score = self._score(current_text)
        
        # 2. Hill Climbing Loop
        # We assume the key is 'stuck' in a local maximum, we try swapping 
        # letters to see if readability improves.
        parent_key = list(best_key)
        
        for _ in range(steps):
            # Mutate: swap two random letters in the key
            child_key = list(parent_key)
            a, b = random.sample(range(26), 2)
            child_key[a], child_key[b] = child_key[b], child_key[a]
            
            # Decrypt and assess
            decrypted_attempt = self._decrypt(msg_upper, child_key)
            score = self._score(decrypted_attempt)
            
            # If improved, adopt the new key
            if score > best_score:
                best_score = score
                best_key = child_key
                parent_key = child_key # Move 'parent' to new best location
        
        # 3. Finalize Output
        # Create final map that handles both Upper and Lower case
        final_plain_upper = "".join(best_key)
        final_plain_lower = final_plain_upper.lower()
        
        # Input alphabet: ABC...Z abc...z
        input_alpha = "".join(self.alphabet) + "".join(self.alphabet).lower()
        # Output alphabet: (Decrypted Upper) (Decrypted Lower)
        output_alpha = final_plain_upper + final_plain_lower
        
        final_trans = str.maketrans(input_alpha, output_alpha)
        final_text = ciphertext.translate(final_trans)
        
        # Create Dictionary {CipherChar: PlainChar}
        solution_dict = {self.alphabet[i]: best_key[i] for i in range(26)}
        
        return final_text, solution_dict

# --- User Interface ---
if __name__ == "__main__":
    print("=== Substitution Cipher Solver ===")
    print("Paste your encrypted message below (press Enter twice to finish):")
    
    lines = []
    while True:
        try:
            line = input()
            if not line: break
            lines.append(line)
        except EOFError:
            break
            
    encrypted_input = "\n".join(lines)
    
    if len(encrypted_input.strip()) < 10:
        print("\n[Warning] Text is very short. Statistical analysis works best on longer sentences.")
        # We proceed anyway, but accuracy might be low.

    solver = SubstitutionCipherSolver()
    print("\nProcessing... (Testing 3000 variations)")
    
    decrypted_text, key_map = solver.solve(encrypted_input)
    
    print("-" * 40)
    print("DECRYPTED MESSAGE:")
    print("-" * 40)
    print(decrypted_text)
    print("-" * 40)
    
    print("\nDETECTED KEY (Cipher -> Plain):")
    # Print dictionary in a readable sorted format
    sorted_key = sorted(key_map.items())
    
    # Format as 4 columns
    for i in range(0, len(sorted_key), 4):
        chunk = sorted_key[i:i+4]
        print(" | ".join([f"{k} -> {v}" for k,v in chunk]))