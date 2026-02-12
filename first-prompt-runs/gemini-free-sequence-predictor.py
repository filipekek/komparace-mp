import math
from typing import List, Tuple, Optional, Union

class SequencePredictor:
    """
    A class to analyze mathematical sequences, predict future terms,
    and determine the underlying formula.
    """

    def __init__(self, data: List[Union[int, float]]):
        if not isinstance(data, list) or len(data) < 3:
            raise ValueError("Input must be a list containing at least 3 numbers.")
        
        # Convert to float for calculation precision, but keep original for display if int
        self.original_data = data
        self.data = [float(x) for x in data]
        self.tolerance = 1e-9 # Tolerance for floating point comparisons

    def _is_approx_equal(self, a: float, b: float) -> bool:
        """Helper to check floating point equality."""
        return math.isclose(a, b, abs_tol=self.tolerance)

    def _format_number(self, num: float) -> Union[int, float]:
        """Converts float to int if it's effectively an integer."""
        if self._is_approx_equal(num, round(num)):
            return int(round(num))
        return round(num, 4)

    def _get_diffs(self, seq: List[float]) -> List[float]:
        """Calculates the first differences of a sequence."""
        return [seq[i+1] - seq[i] for i in range(len(seq) - 1)]

    def analyze(self) -> dict:
        """
        Main driver function. Tries to match specific patterns in order of complexity.
        Returns a dictionary with type, equation, and next values.
        """
        
        # 1. Check Arithmetic (Polynomial Degree 1)
        # 2. Check Geometric
        # 3. Check Fibonacci-style
        # 4. Check Polynomial (Degree 2+)
        
        strategies = [
            self._check_arithmetic,
            self._check_geometric,
            self._check_fibonacci,
            self._check_polynomial_degrees
        ]

        for strategy in strategies:
            result = strategy()
            if result:
                return result

        return {
            "type": "Unknown / Complex",
            "equation": "N/A",
            "next_5_terms": []
        }

    # --- Strategy Implementations ---

    def _check_arithmetic(self) -> Optional[dict]:
        """Checks if sequence is Arithmetic (constant difference)."""
        diffs = self._get_diffs(self.data)
        if not diffs: return None
        
        first_diff = diffs[0]
        if all(self._is_approx_equal(d, first_diff) for d in diffs):
            # Predict
            last_val = self.data[-1]
            next_vals = [self._format_number(last_val + first_diff * i) for i in range(1, 6)]
            
            # Format Equation: a_n = a1 + (n-1)d
            a1 = self._format_number(self.data[0])
            d_fmt = self._format_number(first_diff)
            
            # Simplify equation string
            if d_fmt == 0: eq = f"a_n = {a1}"
            else: 
                sign = "+" if first_diff >= 0 else "-"
                eq = f"a_n = {a1} {sign} {abs(first_diff)}(n-1)"

            return {
                "type": "Arithmetic",
                "equation": eq,
                "next_5_terms": next_vals
            }
        return None

    def _check_geometric(self) -> Optional[dict]:
        """Checks if sequence is Geometric (constant ratio)."""
        if any(self._is_approx_equal(x, 0) for x in self.data):
            return None # Cannot divide by zero

        ratios = [self.data[i+1] / self.data[i] for i in range(len(self.data) - 1)]
        first_ratio = ratios[0]
        
        if all(self._is_approx_equal(r, first_ratio) for r in ratios):
            # Predict
            last_val = self.data[-1]
            next_vals = []
            current = last_val
            for _ in range(5):
                current *= first_ratio
                next_vals.append(self._format_number(current))

            # Format Equation: a_n = a1 * r^(n-1)
            a1 = self._format_number(self.data[0])
            r_fmt = self._format_number(first_ratio)
            eq = f"a_n = {a1} * ({r_fmt})^(n-1)"

            return {
                "type": "Geometric",
                "equation": eq,
                "next_5_terms": next_vals
            }
        return None

    def _check_fibonacci(self) -> Optional[dict]:
        """Checks if sequence follows a_n = a_{n-1} + a_{n-2} logic."""
        # Must verify for all items starting from index 2
        is_fib = True
        for i in range(2, len(self.data)):
            if not self._is_approx_equal(self.data[i], self.data[i-1] + self.data[i-2]):
                is_fib = False
                break
        
        if is_fib:
            next_vals = []
            buffer = list(self.data)
            for _ in range(5):
                next_val = buffer[-1] + buffer[-2]
                buffer.append(next_val)
                next_vals.append(self._format_number(next_val))
            
            return {
                "type": "Fibonacci-like",
                "equation": "a_n = a_{n-1} + a_{n-2}",
                "next_5_terms": next_vals
            }
        return None

    def _check_polynomial_degrees(self) -> Optional[dict]:
        """
        Uses Method of Finite Differences to detect Quadratic, Cubic, etc.
        If the Nth difference is constant, it is a polynomial of degree N.
        """
        current_diffs = self.data
        depth = 0
        max_depth = len(self.data) - 1 # Cannot go deeper than data allows

        while depth < max_depth:
            current_diffs = self._get_diffs(current_diffs)
            depth += 1
            
            # If we run out of data or list is empty
            if not current_diffs: break

            # Check if all diffs in this row are equal
            first = current_diffs[0]
            if all(self._is_approx_equal(x, first) for x in current_diffs):
                # Found a constant difference at 'depth'
                return self._predict_polynomial(depth)
        
        return None

    def _predict_polynomial(self, degree: int) -> dict:
        """
        Extrapolates polynomial sequence and attempts to fit a Quadratic equation
        if degree == 2. (Higher degrees return generic strings).
        """
        # 1. Extrapolate Next 5 Terms using the Difference Table
        # We rebuild the diff table to the constant row, then work backwards
        table = [list(self.data)]
        for _ in range(degree):
            table.append(self._get_diffs(table[-1]))
        
        # Extend the constant row (last row of table)
        constant_diff = table[-1][0]
        predictions = []
        
        # We need to simulate adding 5 new columns to the right
        # Make deep copy to avoid modifying state
        temp_table = [row[:] for row in table]
        
        for _ in range(5):
            # Append constant to the bottom row
            temp_table[-1].append(constant_diff)
            
            # Sum upwards
            for i in range(degree - 1, -1, -1):
                new_val = temp_table[i][-1] + temp_table[i+1][-1]
                temp_table[i].append(new_val)
            
            predictions.append(self._format_number(temp_table[0][-1]))

        # 2. Determine Equation String
        type_name = "Quadratic" if degree == 2 else f"Polynomial (Degree {degree})"
        eq_str = f"Polynomial of order {degree}"
        
        if degree == 2:
            eq_str = self._solve_quadratic_equation()

        return {
            "type": type_name,
            "equation": eq_str,
            "next_5_terms": predictions
        }

    def _solve_quadratic_equation(self) -> str:
        """
        Solves an^2 + bn + c for the first 3 points.
        Using system of equations for n=1, 2, 3:
        1) a + b + c = y0
        2) 4a + 2b + c = y1
        3) 9a + 3b + c = y2
        """
        y0, y1, y2 = self.data[0], self.data[1], self.data[2]
        
        # Second difference is 2a
        diff1 = [y1 - y0, y2 - y1]
        diff2 = diff1[1] - diff1[0]
        
        a = diff2 / 2
        b = (y1 - y0) - 3*a  # Derived from standard form
        c = y0 - a - b
        
        # Formatting
        def fmt(n, is_first=False):
            n = self._format_number(n)
            if n == 0: return ""
            val_str = str(abs(n))
            if n == 1 and not is_first and isinstance(n, int): val_str = "" # Hide 1 in '1n'
            
            sign = " + " if n > 0 else " - "
            if is_first: sign = "" if n > 0 else "-"
            return f"{sign}{val_str}"

        # Construct: an^2 + bn + c
        term_a = fmt(a, True) + "n^2" if a != 0 else ""
        term_b = fmt(b) + "n" if b != 0 else ""
        term_c = fmt(c) if c != 0 else ""
        
        equation = f"a_n = {term_a}{term_b}{term_c}".strip()
        return equation

# --- Usage Helper ---
def solve_sequence(numbers):
    try:
        predictor = SequencePredictor(numbers)
        result = predictor.analyze()
        
        print("-" * 40)
        print(f"Input: {numbers}")
        print(f"Type: {result['type']}")
        print(f"Equation: {result['equation']}")
        print(f"Next 5: {result['next_5_terms']}")
        print("-" * 40)
    except Exception as e:
        print(f"Error: {e}")

# --- Test Cases ---
if __name__ == "__main__":
    # Arithmetic
    solve_sequence([2, 4, 12, 48])
    
    # Geometric
    solve_sequence([3, 9, 27, 81])
    
    # Quadratic (n^2)
    solve_sequence([1, 4, 9, 16])
    
    # Quadratic (Custom: 2n^2 + n + 5) -> 8, 15, 26
    solve_sequence([8, 15, 26, 41])
    
    # Fibonacci
    solve_sequence([1, 1, 2, 3, 5, 8])
    
    # Cubic (n^3)
    solve_sequence([1, 8, 27, 64])