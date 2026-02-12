import numpy as np
import sys
import re
import time

def get_user_input():
    """
    Handles user input, ensuring robust parsing of numbers 
    separated by spaces or commas.
    """
    print("--- Sequence Predictor ---")
    while True:
        user_in = input("Input at least 3 numerals separated by comma or space:\n> ").strip()
        
        # Replace commas with spaces and split
        clean_input = re.sub(r'[,;]', ' ', user_in)
        parts = clean_input.split()
        
        if len(parts) < 3:
            print("Error: Please enter at least 3 numbers to establish a pattern.")
            continue
            
        try:
            # Convert to numpy array of floats for precision
            nums = np.array([float(x) for x in parts])
            
            # Check if inputs are actually integers (for cleaner output later)
            if np.all(np.mod(nums, 1) == 0):
                return nums, True # True indicates they were input as integers
            return nums, False
        except ValueError:
            print("Error: Input contains non-numeric characters. Please try again.")

def format_number(num, is_integer_mode):
    """
    Helper to format numbers cleanly (e.g., 5 instead of 5.0)
    """
    if np.isclose(num, round(num)) and is_integer_mode:
        return str(int(round(num)))
    return f"{num:.4g}"

def check_arithmetic(nums, is_int_mode):
    """
    Checks for Arithmetic Progression: a(n) = a1 + (n-1)d
    """
    diffs = np.diff(nums)
    
    # Check if all differences are close to the first difference
    if np.allclose(diffs, diffs[0]):
        d = diffs[0]
        a1 = nums[0]
        
        # Calculate next 5
        last_val = nums[-1]
        next_vals = [last_val + d * i for i in range(1, 6)]
        
        # Equation string construction
        # a(n) = a1 + dn - d -> dn + (a1 - d)
        intercept = a1 - d
        d_str = format_number(d, is_int_mode)
        int_str = format_number(abs(intercept), is_int_mode)
        sign = "+" if intercept >= 0 else "-"
        
        eq = f"a(n) = {d_str}n {sign} {int_str}"
        return "Arithmetic", eq, next_vals
    return None

def check_geometric(nums, is_int_mode):
    """
    Checks for Geometric Progression: a(n) = a1 * r^(n-1)
    """
    if 0 in nums: return None # Avoid division by zero
    
    ratios = nums[1:] / nums[:-1]
    
    if np.allclose(ratios, ratios[0]):
        r = ratios[0]
        a1 = nums[0]
        
        # Calculate next 5
        last_val = nums[-1]
        next_vals = [last_val * (r ** i) for i in range(1, 6)]
        
        r_str = format_number(r, is_int_mode)
        a1_str = format_number(a1, is_int_mode)
        
        eq = f"a(n) = {a1_str} * ({r_str})^(n-1)"
        return "Geometric", eq, next_vals
    return None

def check_fibonacci_like(nums, is_int_mode):
    """
    Checks for pattern a(n) = a(n-1) + a(n-2)
    """
    is_fib = True
    for i in range(2, len(nums)):
        if not np.isclose(nums[i], nums[i-1] + nums[i-2]):
            is_fib = False
            break
            
    if is_fib:
        next_vals = []
        curr = list(nums)
        for _ in range(5):
            next_val = curr[-1] + curr[-2]
            next_vals.append(next_val)
            curr.append(next_val)
            
        return "Fibonacci-like", "a(n) = a(n-1) + a(n-2)", next_vals
    return None

def check_polynomial(nums, is_int_mode):
    """
    Checks for polynomial sequences using numpy polyfit.
    Iterates degrees from 2 up to len(nums)-1.
    """
    x = np.arange(1, len(nums) + 1) # Position 1-based indexing
    
    # We try fitting polynomials of increasing degree
    # We stop if the error is negligible
    max_degree = min(len(nums) - 1, 5) # Cap at degree 5 to prevent overfitting on tiny sets
    
    for deg in range(2, max_degree + 1):
        coeffs = np.polyfit(x, nums, deg)
        
        # Check valid fit
        predicted = np.polyval(coeffs, x)
        if np.allclose(nums, predicted, atol=1e-4):
            
            # Predict next 5
            next_x = np.arange(len(nums) + 1, len(nums) + 6)
            next_vals = np.polyval(coeffs, next_x)
            
            # Build Equation String
            eq_parts = []
            for i, c in enumerate(coeffs):
                power = deg - i
                if np.isclose(c, 0, atol=1e-5): continue
                
                c_val = format_number(abs(c), is_int_mode)
                sign = " - " if c < 0 else " + "
                if i == 0 and c >= 0: sign = "" # First term positive
                if i == 0 and c < 0: sign = "-" # First term negative
                
                term = ""
                if power > 1: term = f"{c_val}n^{power}"
                elif power == 1: term = f"{c_val}n"
                else: term = f"{c_val}"
                
                eq_parts.append(f"{sign}{term}")
            
            eq_str = "a(n) = " + "".join(eq_parts).strip().lstrip("+ ")
            seq_type = "Quadratic" if deg == 2 else "Cubic" if deg == 3 else f"Polynomial (Deg {deg})"
            
            return seq_type, eq_str, next_vals
            
    return None

def main():
    nums, is_int = get_user_input()
    t1 = time.perf_counter()
    # Run Detectors
    result = check_arithmetic(nums, is_int)
    
    if not result:
        result = check_geometric(nums, is_int)
        
    if not result:
        result = check_fibonacci_like(nums, is_int)
        
    if not result:
        result = check_polynomial(nums, is_int)
        
    print("-" * 30)
    
    if result:
        seq_type, eq, next_vals = result
        
        # Format next values output
        next_vals_str = " ".join([format_number(x, is_int) for x in next_vals])
        
        print(f"Type: {seq_type}")
        print(f"Equation: {eq}")
        print(f"Next 5 numbers: {next_vals_str}")
    else:
        print("Pattern Recognition Failed: Sequence too complex or data insufficient.")
    
    print("-" * 30)
    t2 = time.perf_counter()
    print(t2-t1)

if __name__ == "__main__":
    main()