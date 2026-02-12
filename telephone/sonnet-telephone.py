"""
Text Compression and Decompression using Huffman Coding
This implementation provides lossless compression with optimal prefix-free encoding.
"""

import sys
from collections import Counter, deque
import time

class HuffmanNode:
    """Node class for building the Huffman tree."""
    
    def __init__(self, char=None, freq=0, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right
    
    def __lt__(self, other):
        return self.freq < other.freq


class MinHeap:
    """Simple min-heap implementation for Huffman tree construction."""
    
    def __init__(self):
        self.heap = []
    
    def push(self, item):
        self.heap.append(item)
        self._sift_up(len(self.heap) - 1)
    
    def pop(self):
        if len(self.heap) == 0:
            return None
        if len(self.heap) == 1:
            return self.heap.pop()
        
        root = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._sift_down(0)
        return root
    
    def _sift_up(self, idx):
        parent = (idx - 1) // 2
        if idx > 0 and self.heap[idx] < self.heap[parent]:
            self.heap[idx], self.heap[parent] = self.heap[parent], self.heap[idx]
            self._sift_up(parent)
    
    def _sift_down(self, idx):
        left = 2 * idx + 1
        right = 2 * idx + 2
        smallest = idx
        
        if left < len(self.heap) and self.heap[left] < self.heap[smallest]:
            smallest = left
        if right < len(self.heap) and self.heap[right] < self.heap[smallest]:
            smallest = right
        
        if smallest != idx:
            self.heap[idx], self.heap[smallest] = self.heap[smallest], self.heap[idx]
            self._sift_down(smallest)
    
    def __len__(self):
        return len(self.heap)


def build_huffman_tree(text):
    """Build a Huffman tree from the input text."""
    if not text:
        return None
    
    # Count character frequencies
    freq_map = Counter(text)
    
    # Handle single character case
    if len(freq_map) == 1:
        char = list(freq_map.keys())[0]
        return HuffmanNode(char=char, freq=freq_map[char])
    
    # Build min heap
    min_heap = MinHeap()
    for char, freq in freq_map.items():
        min_heap.push(HuffmanNode(char=char, freq=freq))
    
    # Build Huffman tree
    while len(min_heap) > 1:
        left = min_heap.pop()
        right = min_heap.pop()
        
        merged = HuffmanNode(
            freq=left.freq + right.freq,
            left=left,
            right=right
        )
        min_heap.push(merged)
    
    return min_heap.pop()


def build_codes(root):
    """Build Huffman codes from the tree."""
    if root is None:
        return {}
    
    # Handle single character case
    if root.char is not None and root.left is None and root.right is None:
        return {root.char: '0'}
    
    codes = {}
    
    def traverse(node, code):
        if node is None:
            return
        
        if node.char is not None:
            codes[node.char] = code
            return
        
        traverse(node.left, code + '0')
        traverse(node.right, code + '1')
    
    traverse(root, '')
    return codes


def compress(text):
    """Compress the input text using Huffman coding."""
    if not text:
        return "", {}, ""
    
    # Build Huffman tree and codes
    root = build_huffman_tree(text)
    codes = build_codes(root)
    
    # Encode the text
    encoded = ''.join(codes[char] for char in text)
    
    # Convert binary string to bytes for storage efficiency
    # Pad to make length multiple of 8
    padding = (8 - len(encoded) % 8) % 8
    encoded_padded = encoded + '0' * padding
    
    # Convert to bytes
    compressed_bytes = bytearray()
    for i in range(0, len(encoded_padded), 8):
        byte = encoded_padded[i:i+8]
        compressed_bytes.append(int(byte, 2))
    
    # Store padding info
    compressed_data = bytes([padding]) + bytes(compressed_bytes)
    
    return compressed_data, codes, encoded


def decompress(compressed_data, codes):
    """Decompress the compressed data using Huffman codes."""
    if not compressed_data or not codes:
        return ""
    
    # Reverse the codes dictionary
    reverse_codes = {code: char for char, code in codes.items()}
    
    # Extract padding and compressed bytes
    padding = compressed_data[0]
    compressed_bytes = compressed_data[1:]
    
    # Convert bytes back to binary string
    binary_string = ''.join(format(byte, '08b') for byte in compressed_bytes)
    
    # Remove padding
    if padding > 0:
        binary_string = binary_string[:-padding]
    
    # Decode using Huffman codes
    decoded = []
    current_code = ""
    
    for bit in binary_string:
        current_code += bit
        if current_code in reverse_codes:
            decoded.append(reverse_codes[current_code])
            current_code = ""
    
    return ''.join(decoded)


def calculate_compression_ratio(original_size, compressed_size):
    """Calculate compression ratio as percentage."""
    if original_size == 0:
        return 0.0
    return (compressed_size / original_size) * 100


def calculate_accuracy(original, decompressed):
    """Calculate decompression accuracy as percentage."""
    if not original:
        return 100.0 if not decompressed else 0.0
    
    if original == decompressed:
        return 100.0
    
    # Character-level accuracy
    matches = sum(1 for o, d in zip(original, decompressed) if o == d)
    max_len = max(len(original), len(decompressed))
    return (matches / max_len) * 100


def main():
    """Main function to run the compression/decompression demo."""
    print("=" * 60)
    print("Text Compression and Decompression System")
    print("Using Huffman Coding Algorithm")
    print("=" * 60)
    print()
    
    # Get input from user
    print("Enter the message to compress:")
    message = input("> ")
    t1 = time.perf_counter()
    
    # Handle empty input
    if not message:
        print("\nError: Empty message provided. Cannot compress.")
        return
    
    print("\n" + "-" * 60)
    print("COMPRESSION PROCESS")
    print("-" * 60)
    
    # Compress
    compressed_data, codes, encoded_binary = compress(message)
    
    # Calculate sizes
    original_size = len(message.encode('utf-8'))  # Size in bytes
    compressed_size = len(compressed_data)  # Size in bytes
    
    # Display Huffman codes
    print("\nHuffman Codes Generated:")
    for char, code in sorted(codes.items(), key=lambda x: len(x[1])):
        display_char = repr(char) if char in ['\n', '\t', ' '] else char
        print(f"  {display_char}: {code}")
    
    # Display compressed message
    print(f"\nCompressed Data (hex): {compressed_data.hex()}")
    print(f"Binary Encoding Preview: {encoded_binary[:100]}{'...' if len(encoded_binary) > 100 else ''}")
    
    print("\n" + "-" * 60)
    print("DECOMPRESSION PROCESS")
    print("-" * 60)
    
    # Decompress
    decompressed_message = decompress(compressed_data, codes)
    
    # Calculate metrics
    compression_ratio = calculate_compression_ratio(original_size, compressed_size)
    accuracy = calculate_accuracy(message, decompressed_message)
    t2 = time.perf_counter()
    print(t2-t1)

    # Display results
    print(f"\nDecompressed Message: {repr(decompressed_message)}")
    
    print("\n" + "=" * 60)
    print("COMPRESSION STATISTICS")
    print("=" * 60)
    print(f"Original Message Length:    {len(message)} characters")
    print(f"Original Size:              {original_size} bytes")
    print(f"Compressed Size:            {compressed_size} bytes")
    print(f"Size Reduction:             {original_size - compressed_size} bytes")
    print(f"Compression Ratio:          {compression_ratio:.2f}%")
    print(f"Space Saved:                {100 - compression_ratio:.2f}%")
    print(f"Decompression Accuracy:     {accuracy:.2f}%")
    
    # Validation
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)
    
    if accuracy == 100.0:
        print("✓ Decompression is LOSSLESS (100% accurate)")
    else:
        print("✗ Decompression is LOSSY")
        print(f"  Accuracy: {accuracy:.2f}%")
    
    if compressed_size < original_size:
        print("✓ Compression SUCCESSFUL (compressed size < original size)")
    elif compressed_size == original_size:
        print("⚠ Compression NEUTRAL (compressed size = original size)")
    else:
        print("✗ Compression FAILED (compressed size > original size)")
        print("  Note: This can happen with very short messages or high entropy data")
    
    print("=" * 60)
    

if __name__ == "__main__":
    try:
        
        main()
        
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError occurred: {e}")
        sys.exit(1)