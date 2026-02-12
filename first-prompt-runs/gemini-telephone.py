import math

class LossyHalvingCompressor:
    """
    A compressor that strictly halves the size of a string by averaging 
    adjacent character ASCII values. This is a lossy compression algorithm.
    """

    def compress(self, message: str) -> str:
        """
        Compresses the message by averaging every pair of characters.
        Resulting size is ceil(len(message) / 2).
        """
        if not message:
            return ""

        compressed_chars = []
        length = len(message)

        # Iterate through the message in steps of 2
        for i in range(0, length - 1, 2):
            char1 = message[i]
            char2 = message[i+1]
            
            # Calculate the average ASCII value
            avg_val = (ord(char1) + ord(char2)) // 2
            compressed_chars.append(chr(avg_val))

        # Handle the edge case of an odd-length string
        if length % 2 != 0:
            compressed_chars.append(message[-1])

        return "".join(compressed_chars)

    def decompress(self, compressed_message: str, force_odd_length: bool = False) -> str:
        """
        Attempts to reconstruct the original message by duplicating characters.
        
        Args:
            compressed_message: The compressed string.
            force_odd_length: If True, assumes the original was odd length and 
                              does not duplicate the final character.
        """
        if not compressed_message:
            return ""

        decompressed_chars = []
        length = len(compressed_message)

        # Iterate through all characters except possibly the last one
        limit = length - 1 if force_odd_length else length
        
        for i in range(limit):
            char = compressed_message[i]
            # Naive reconstruction: duplicate the average character
            decompressed_chars.append(char)
            decompressed_chars.append(char)

        # Handle the last character if we know the original was odd
        if force_odd_length:
            decompressed_chars.append(compressed_message[-1])

        return "".join(decompressed_chars)

    def calculate_accuracy(self, original: str, reconstructed: str) -> float:
        """
        Calculates the character matching accuracy between original and reconstructed strings.
        Returns a percentage (0.0 to 100.0).
        """
        if not original:
            return 100.0 if not reconstructed else 0.0

        # Determine the length to compare (minimum of the two to avoid index errors)
        min_len = min(len(original), len(reconstructed))
        matches = 0

        for i in range(min_len):
            if original[i] == reconstructed[i]:
                matches += 1

        # Accuracy is matches / original length
        return (matches / len(original)) * 100

def run_demo(message: str):
    compressor = LossyHalvingCompressor()
    
    print(f"--- Processing: '{message}' ---")
    
    # 1. Compress
    compressed = compressor.compress(message)
    print(f"Original Size:   {len(message)}")
    print(f"Compressed Size: {len(compressed)}")
    print(f"Compressed Content (Repr): {repr(compressed)}")

    # 2. Decompress
    # We pass a hint about odd length to ensure size matches exactly for the demo
    is_odd = len(message) % 2 != 0
    reconstructed = compressor.decompress(compressed, force_odd_length=is_odd)
    print(f"Reconstructed:   '{reconstructed}'")

    # 3. Check Accuracy
    accuracy = compressor.calculate_accuracy(message, reconstructed)
    print(f"Accuracy:        {accuracy:.2f}%")
    print("")

if __name__ == "__main__":
    # Test Case 1: Repeating characters (High Accuracy)
    run_demo("""
    "Foxes are small wild animals found in many parts of the world. They live in forests, fields, mountains, and even near cities. Foxes have soft fur, long tails, and sharp ears. Most foxes are red, but some are gray, white, or brown. They are clever and careful animals. Foxes usually hunt at night. They eat mice, birds, insects, fruit, and plants. Foxes live alone or in small families. They use dens in the ground for rest and safety. People often see foxes as symbols of intelligence and adaptability. Their survival skills help them live in changing environments around humans today.
""")