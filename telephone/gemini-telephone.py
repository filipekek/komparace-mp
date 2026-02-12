import sys
import io
from difflib import SequenceMatcher
import time

class AdaptiveCompressor:
    """
    A compressor that attempts lossless compression (LZW) first.
    If the size constraint (compressed < original) is not met, 
    it falls back to lossy compression (character dropping) to satisfy the requirement.
    """

    def __init__(self):
        # Maximum unicode point for mapping codes to characters
        self.MAX_UNICODE = 0x10FFFF 

    def _lzw_compress(self, uncompressed):
        """
        Standard LZW compression algorithm.
        Returns a string where each character represents a dictionary code.
        """
        # Initialize dictionary with ASCII/Unicode basics (0-255)
        dict_size = 256
        dictionary = {chr(i): i for i in range(dict_size)}
        
        w = ""
        result = []
        
        for c in uncompressed:
            wc = w + c
            if wc in dictionary:
                w = wc
            else:
                result.append(dictionary[w])
                # Add wc to the dictionary
                if dict_size < self.MAX_UNICODE:
                    dictionary[wc] = dict_size
                    dict_size += 1
                w = c
        
        if w:
            result.append(dictionary[w])
            
        # Convert integer codes to a unicode string to mimic a 'compressed file'
        try:
            return "".join(chr(code) for code in result)
        except (ValueError, OverflowError):
            # In the extremely rare case dictionary grows too large for chr()
            return uncompressed + "FAIL"

    def _lzw_decompress(self, compressed):
        """
        Standard LZW decompression algorithm.
        """
        dict_size = 256
        dictionary = {i: chr(i) for i in range(dict_size)}
        
        # Convert unicode string back to integer codes
        codes = [ord(c) for c in compressed]
        
        if not codes:
            return ""

        # Initialize with the first character
        w = chr(codes.pop(0))
        result = io.StringIO()
        result.write(w)
        
        for k in codes:
            if k in dictionary:
                entry = dictionary[k]
            elif k == dict_size:
                # Special LZW edge case: cScSc
                entry = w + w[0]
            else:
                raise ValueError(f"Bad compressed code: {k}")
            
            result.write(entry)
            
            # Add w + entry[0] to the dictionary
            if dict_size < self.MAX_UNICODE:
                dictionary[dict_size] = w + entry[0]
                dict_size += 1
            
            w = entry
            
        return result.getvalue()

    def process(self, original_msg):
        """
        Orchestrates the compression and decompression cycle.
        Decides between Lossless and Lossy based on size constraints.
        """
        if not original_msg:
            return {
                "original": "",
                "compressed": "",
                "decompressed": "",
                "method": "None",
                "accuracy": 100.0
            }

        # 1. Attempt Lossless LZW
        compressed_lzw = self._lzw_compress(original_msg)
        
        # 2. Check Constraint: Compressed size MUST be smaller than original
        if len(compressed_lzw) < len(original_msg):
            # Success with Lossless
            decompressed = self._lzw_decompress(compressed_lzw)
            method = "LZW (Lossless)"
            final_compressed = compressed_lzw
        else:
            # 3. Fallback to Lossy if LZW didn't shrink the message
            # Method: Strip vowels and non-alphanumeric chars until smaller
            method = "Vowel/Char Drop (Lossy)"
            vowels = set("aeiouAEIOU")
            
            # Create a generator that filters chars
            filtered_chars = [c for c in original_msg if c not in vowels]
            final_compressed = "".join(filtered_chars)

            # If still not smaller (e.g. string was "fly"), force truncation
            if len(final_compressed) >= len(original_msg):
                # Simply cut the string to ensure it is strictly smaller
                # If len is 1, it becomes empty string.
                final_compressed = original_msg[:-1]
                method = "Forced Truncation (Lossy)"

            # "Decompression" for lossy is just returning what we have
            # (We cannot reconstruct lost data without a massive external dict)
            decompressed = final_compressed

        # Calculate Accuracy
        accuracy = SequenceMatcher(None, original_msg, decompressed).ratio() * 100

        return {
            "original": original_msg,
            "compressed": final_compressed,
            "decompressed": decompressed,
            "method": method,
            "accuracy": accuracy
        }

def main():
    print("--- Adaptive Text Compressor ---")
    print("Goal: Minimize size while maximizing accuracy.")
    print("Constraint: Compressed size must ALWAYS be smaller than original.\n")

    try:
        user_input = input("Enter message to compress: ")
    except (EOFError, KeyboardInterrupt):
        sys.exit("\nInput cancelled.")

    if not user_input:
        print("Error: Message cannot be empty.")
        return
    t1 = time.perf_counter()

    # Initialize and run
    compressor = AdaptiveCompressor()
    result = compressor.process(user_input)

    # Output Formatting
    # Use repr() to visualize control characters/spaces safely
    orig_len = len(result['original'])
    comp_len = len(result['compressed'])
    
    # Avoid division by zero
    reduction = (1 - (comp_len / orig_len)) * 100 if orig_len > 0 else 0

    print("\n" + "="*40)
    print(f"Algorithm Selected:  {result['method']}")
    print("="*40)
    print(f"Original Message:    {repr(result['original'])}")
    print(f"Compressed Message:  {repr(result['compressed'])}")
    print(f"Decompressed Result: {repr(result['decompressed'])}")
    print("-" * 40)
    print(f"Original Size:       {orig_len} chars")
    print(f"Compressed Size:     {comp_len} chars")
    print(f"Size Reduction:      {reduction:.2f}%")
    print(f"Decompression Accuracy: {result['accuracy']:.2f}%")
    print("="*40)
    t2 = time.perf_counter()
    print(t2-t1)

if __name__ == "__main__":
    
    main()
    