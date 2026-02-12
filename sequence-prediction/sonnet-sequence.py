import numpy as np
from typing import List, Tuple, Optional
import time

def parse_input(user_input: str) -> List[float]:
    """Parse user input, handling both comma and space separators."""
    # Replace commas with spaces and split
    cleaned = user_input.replace(',', ' ')
    try:
        numbers = [float(x) for x in cleaned.split() if x.strip()]
        return numbers
    except ValueError:
        raise ValueError("Invalid input. Please enter numbers separated by commas or spaces.")

def check_arithmetic(sequence: List[float]) -> Optional[Tuple[str, List[float]]]:
    """Check if sequence is arithmetic (constant difference)."""
    if len(sequence) < 2:
        return None
    
    differences = [sequence[i+1] - sequence[i] for i in range(len(sequence)-1)]
    
    # Check if all differences are approximately equal
    if all(abs(d - differences[0]) < 1e-9 for d in differences):
        d = differences[0]
        a0 = sequence[0]
        
        # Generate equation: a(n) = a0 + d*n (where n starts from 0)
        # Or in form: a(n) = d*n + a0
        next_values = [sequence[-1] + d * (i+1) for i in range(5)]
        
        if abs(d) < 1e-9:
            equation = f"a(n) = {format_number(a0)}"
        elif abs(a0) < 1e-9:
            equation = f"a(n) = {format_number(d)}n"
        else:
            equation = f"a(n) = {format_number(d)}n + {format_number(a0)}"
        
        return ("Arithmetic", equation, next_values)
    
    return None

def check_geometric(sequence: List[float]) -> Optional[Tuple[str, List[float]]]:
    """Check if sequence is geometric (constant ratio)."""
    if len(sequence) < 2:
        return None
    
    # Check for zeros which would break geometric sequence
    if any(abs(x) < 1e-9 for x in sequence[:-1]):
        return None
    
    ratios = [sequence[i+1] / sequence[i] for i in range(len(sequence)-1)]
    
    if all(abs(r - ratios[0]) < 1e-9 for r in ratios):
        r = ratios[0]
        a0 = sequence[0]
        
        next_values = [sequence[-1] * (r ** (i+1)) for i in range(5)]
        
        if abs(r - 1) < 1e-9:
            equation = f"a(n) = {format_number(a0)}"
        else:
            equation = f"a(n) = {format_number(a0)} × {format_number(r)}^n"
        
        return ("Geometric", equation, next_values)
    
    return None

def check_quadratic(sequence: List[float]) -> Optional[Tuple[str, List[float]]]:
    """Check if sequence follows a quadratic pattern (second differences constant)."""
    if len(sequence) < 3:
        return None
    
    first_diff = [sequence[i+1] - sequence[i] for i in range(len(sequence)-1)]
    
    if len(first_diff) < 2:
        return None
    
    second_diff = [first_diff[i+1] - first_diff[i] for i in range(len(first_diff)-1)]
    
    if all(abs(d - second_diff[0]) < 1e-9 for d in second_diff):
        # Fit quadratic: a(n) = An² + Bn + C
        # Use least squares fitting with numpy
        n_values = np.arange(len(sequence))
        
        # Create design matrix for quadratic fit
        A_matrix = np.column_stack([n_values**2, n_values, np.ones(len(sequence))])
        coeffs = np.linalg.lstsq(A_matrix, sequence, rcond=None)[0]
        
        a, b, c = coeffs
        
        # Generate next 5 values
        next_n = np.arange(len(sequence), len(sequence) + 5)
        next_values = [a * n**2 + b * n + c for n in next_n]
        
        # Format equation
        terms = []
        if abs(a) > 1e-9:
            terms.append(f"{format_number(a)}n²")
        if abs(b) > 1e-9:
            terms.append(f"{format_number(b, sign=True)}n" if terms else f"{format_number(b)}n")
        if abs(c) > 1e-9 or not terms:
            terms.append(f"{format_number(c, sign=True)}" if terms else f"{format_number(c)}")
        
        equation = f"a(n) = {' '.join(terms)}" if terms else "a(n) = 0"
        
        return ("Quadratic", equation, next_values)
    
    return None

def check_fibonacci_like(sequence: List[float]) -> Optional[Tuple[str, List[float]]]:
    """Check if sequence follows Fibonacci-like pattern (each term = sum of previous two)."""
    if len(sequence) < 3:
        return None
    
    is_fibonacci = all(
        abs(sequence[i] - (sequence[i-1] + sequence[i-2])) < 1e-9 
        for i in range(2, len(sequence))
    )
    
    if is_fibonacci:
        next_values = []
        current = sequence[-2:]
        for _ in range(5):
            next_val = current[-1] + current[-2]
            next_values.append(next_val)
            current = [current[-1], next_val]
        
        equation = "a(n) = a(n-1) + a(n-2)"
        return ("Fibonacci-like", equation, next_values)
    
    return None

def check_polynomial(sequence: List[float], max_degree: int = 5) -> Optional[Tuple[str, List[float]]]:
    """Check if sequence follows a polynomial pattern using numpy polyfit."""
    if len(sequence) < 2:
        return None
    
    n_values = np.arange(len(sequence))
    
    # Try different polynomial degrees
    for degree in range(3, min(max_degree + 1, len(sequence))):
        coeffs = np.polyfit(n_values, sequence, degree)
        poly = np.poly1d(coeffs)
        
        # Check if polynomial fits well
        predicted = poly(n_values)
        if np.allclose(predicted, sequence, rtol=1e-9, atol=1e-9):
            # Generate next 5 values
            next_n = np.arange(len(sequence), len(sequence) + 5)
            next_values = poly(next_n).tolist()
            
            # Format equation
            terms = []
            for i, coeff in enumerate(coeffs):
                power = degree - i
                if abs(coeff) > 1e-9:
                    if power == 0:
                        terms.append(f"{format_number(coeff, sign=bool(terms))}")
                    elif power == 1:
                        terms.append(f"{format_number(coeff, sign=bool(terms))}n")
                    else:
                        terms.append(f"{format_number(coeff, sign=bool(terms))}n^{power}")
            
            equation = f"a(n) = {' '.join(terms)}" if terms else "a(n) = 0"
            
            return (f"Polynomial (degree {degree})", equation, next_values)
    
    return None

def check_exponential(sequence: List[float]) -> Optional[Tuple[str, List[float]]]:
    """Check if sequence follows exponential pattern: a(n) = A × B^n."""
    if len(sequence) < 3:
        return None
    
    # Check for non-positive values (log won't work)
    if any(x <= 0 for x in sequence):
        return None
    
    # Take log and check if it's linear
    log_seq = np.log(sequence)
    n_values = np.arange(len(sequence))
    
    # Fit line to log values
    coeffs = np.polyfit(n_values, log_seq, 1)
    poly = np.poly1d(coeffs)
    predicted_log = poly(n_values)
    
    if np.allclose(predicted_log, log_seq, rtol=1e-2, atol=1e-2):
        # Extract A and B from log(A × B^n) = log(A) + n×log(B)
        log_B = coeffs[0]
        log_A = coeffs[1]
        A = np.exp(log_A)
        B = np.exp(log_B)
        
        # Generate next 5 values
        next_n = np.arange(len(sequence), len(sequence) + 5)
        next_values = [A * (B ** n) for n in next_n]
        
        equation = f"a(n) = {format_number(A)} × {format_number(B)}^n"
        
        return ("Exponential", equation, next_values)
    
    return None

def format_number(num: float, sign: bool = False) -> str:
    """Format number for display, handling integers and decimals nicely."""
    # Check if it's effectively an integer
    if abs(num - round(num)) < 1e-9:
        num = int(round(num))
        if sign and num >= 0:
            return f"+ {num}"
        return str(num)
    else:
        formatted = f"{num:.6f}".rstrip('0').rstrip('.')
        if sign and num >= 0:
            return f"+ {formatted}"
        return formatted

def predict_sequence(sequence: List[float]) -> Tuple[str, str, List[float]]:
    """Main function to identify sequence type and predict next values."""
    
    # Try different sequence types in order of specificity
    checkers = [
        check_arithmetic,
        check_geometric,
        check_fibonacci_like,
        check_quadratic,
        check_exponential,
        check_polynomial
    ]
    
    for checker in checkers:
        result = checker(sequence)
        if result:
            return result
    
    # If no pattern found, use polynomial fitting as fallback
    return ("Unknown", "Pattern not recognized", [])

def main():
    print("=" * 60)
    print("Mathematical Sequence Predictor")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\nInput at least 3 numbers separated by comma or space: ")
            t1 = time.perf_counter()
            if not user_input.strip():
                print("Please enter some numbers.")
                continue
            
            sequence = parse_input(user_input)
            
            if len(sequence) < 3:
                print("Please enter at least 3 numbers.")
                continue
            
            print(f"\nInput sequence: {' '.join(map(str, sequence))}")
            
            seq_type, equation, next_values = predict_sequence(sequence)
            
            print(f"\nType: {seq_type}")
            print(f"Equation: {equation}")
            
            if next_values:
                formatted_next = ' '.join(format_number(x) for x in next_values)
                print(f"Next 5 numbers: {formatted_next}")
            else:
                print("Could not predict next numbers.")
            
            print("\n" + "=" * 60)
            t2 = time.perf_counter()
            print(t2-t1)
            # Ask if user wants to continue
            continue_choice = input("\nAnalyze another sequence? (y/n): ").strip().lower()
            if continue_choice != 'y':
                print("\nThank you for using the Sequence Predictor!")
                break
                
        except ValueError as e:
            print(f"Error: {e}")
        except KeyboardInterrupt:
            print("\n\nProgram terminated by user.")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()