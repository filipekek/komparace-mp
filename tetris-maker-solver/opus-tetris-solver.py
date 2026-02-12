"""
Tetris-styled Puzzle Solver

This module provides a function to solve tetris-like puzzles where pieces
must be placed into a grid without overlapping fixed pieces or each other.
Pieces can be rotated but not flipped (mirrored).
"""


def solve_puzzle(grid, pieces):
    """
    Solve a tetris-styled puzzle by placing all pieces into the grid.
    
    Args:
        grid: A list of lists representing the puzzle grid. 
              0 represents empty cells, non-zero values represent fixed pieces.
        pieces: A list of pieces, where each piece is a list of lists with 
                1s representing the piece shape and 0s representing empty space.
    
    Returns:
        The number of valid solutions found.
    """
    if not grid or not grid[0]:
        # Empty grid case - valid only if no pieces with cells
        if not pieces or all(_is_empty_piece(p) for p in pieces):
            return 1
        return 0
    
    # Filter out empty pieces
    valid_pieces = [p for p in pieces if not _is_empty_piece(p)]
    
    # Count empty cells in grid
    empty_count = sum(1 for row in grid for cell in row if cell == 0)
    
    if not valid_pieces:
        # No pieces to place - valid only if no empty cells
        return 1 if empty_count == 0 else 0
    
    rows = len(grid)
    cols = len(grid[0])
    
    # Validate grid is rectangular
    for row in grid:
        if len(row) != cols:
            return 0  # Invalid grid
    
    # Create a working copy of the grid
    working_grid = [row[:] for row in grid]
    
    # Find the next available number for labeling pieces
    existing_numbers = set()
    for row in grid:
        for cell in row:
            if cell != 0:
                existing_numbers.add(cell)
    
    # Count empty cells in grid
    empty_cells = sum(1 for row in working_grid for cell in row if cell == 0)
    
    # Count total cells needed by all pieces
    total_piece_cells = sum(_count_piece_cells(p) for p in valid_pieces)
    
    # Quick check: if piece cells don't match empty cells, no solution
    if total_piece_cells != empty_cells:
        return 0
    
    # Generate all rotations for each piece
    piece_rotations = []
    for piece in valid_pieces:
        rotations = _get_all_rotations(piece)
        # Normalize each rotation to get unique shapes
        unique_rotations = _get_unique_rotations(rotations)
        piece_rotations.append(unique_rotations)
    
    # Use backtracking to find all solutions
    solution_count = [0]  # Using list to allow modification in nested function
    
    _backtrack(working_grid, piece_rotations, 0, existing_numbers, solution_count, rows, cols)
    
    return solution_count[0]


def _is_empty_piece(piece):
    """Check if a piece is empty (contains no 1s)."""
    if not piece:
        return True
    for row in piece:
        if not row:
            continue
        for cell in row:
            if cell == 1:
                return False
    return True


def _count_piece_cells(piece):
    """Count the number of filled cells (1s) in a piece."""
    count = 0
    for row in piece:
        for cell in row:
            if cell == 1:
                count += 1
    return count


def _normalize_piece(piece):
    """
    Normalize a piece by removing empty rows/columns and converting to tuple form.
    Returns a tuple of tuples for hashability.
    """
    if not piece or not piece[0]:
        return tuple()
    
    # Find bounding box of the piece
    min_row, max_row = len(piece), -1
    min_col, max_col = len(piece[0]), -1
    
    for i, row in enumerate(piece):
        for j, cell in enumerate(row):
            if cell == 1:
                min_row = min(min_row, i)
                max_row = max(max_row, i)
                min_col = min(min_col, j)
                max_col = max(max_col, j)
    
    if max_row == -1:  # Empty piece
        return tuple()
    
    # Extract the normalized piece
    normalized = []
    for i in range(min_row, max_row + 1):
        row_data = []
        for j in range(min_col, max_col + 1):
            if j < len(piece[i]):
                row_data.append(piece[i][j])
            else:
                row_data.append(0)
        normalized.append(tuple(row_data))
    
    return tuple(normalized)


def _rotate_90(piece):
    """Rotate a piece 90 degrees clockwise."""
    if not piece or not piece[0]:
        return []
    
    rows = len(piece)
    cols = len(piece[0])
    
    # New dimensions after rotation
    rotated = []
    for j in range(cols):
        new_row = []
        for i in range(rows - 1, -1, -1):
            if j < len(piece[i]):
                new_row.append(piece[i][j])
            else:
                new_row.append(0)
        rotated.append(new_row)
    
    return rotated


def _get_all_rotations(piece):
    """Get all 4 rotations of a piece (0°, 90°, 180°, 270°)."""
    rotations = [piece]
    current = piece
    
    for _ in range(3):
        current = _rotate_90(current)
        rotations.append(current)
    
    return rotations


def _get_unique_rotations(rotations):
    """Filter rotations to keep only unique shapes."""
    seen = set()
    unique = []
    
    for rotation in rotations:
        normalized = _normalize_piece(rotation)
        if normalized and normalized not in seen:
            seen.add(normalized)
            # Convert back to list form for easier manipulation
            unique.append([list(row) for row in normalized])
    
    return unique


def _get_piece_cells(piece):
    """Get list of (row, col) positions of filled cells in a piece."""
    cells = []
    for i, row in enumerate(piece):
        for j, cell in enumerate(row):
            if cell == 1:
                cells.append((i, j))
    return cells


def _can_place_piece(grid, piece, start_row, start_col, rows, cols):
    """Check if a piece can be placed at the given position."""
    piece_cells = _get_piece_cells(piece)
    
    for dr, dc in piece_cells:
        r, c = start_row + dr, start_col + dc
        
        # Check bounds
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return False
        
        # Check if cell is empty
        if grid[r][c] != 0:
            return False
    
    return True


def _place_piece(grid, piece, start_row, start_col, piece_number):
    """Place a piece on the grid with the given number."""
    piece_cells = _get_piece_cells(piece)
    
    for dr, dc in piece_cells:
        r, c = start_row + dr, start_col + dc
        grid[r][c] = piece_number


def _remove_piece(grid, piece, start_row, start_col):
    """Remove a piece from the grid (set cells back to 0)."""
    piece_cells = _get_piece_cells(piece)
    
    for dr, dc in piece_cells:
        r, c = start_row + dr, start_col + dc
        grid[r][c] = 0


def _get_next_number(existing_numbers):
    """Get the next available number for labeling a piece."""
    num = 1
    while num in existing_numbers:
        num += 1
    return num


def _find_first_empty_cell(grid, rows, cols):
    """Find the first empty cell in the grid (reading order)."""
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 0:
                return (i, j)
    return None


def _backtrack(grid, piece_rotations, piece_index, existing_numbers, solution_count, rows, cols):
    """
    Backtracking algorithm to find all valid piece placements.
    
    Uses the strategy of finding the first empty cell and trying to fill it
    with any remaining piece, which helps prune the search space.
    """
    # If all pieces have been placed, we found a solution
    if piece_index >= len(piece_rotations):
        solution_count[0] += 1
        return
    
    # Find the first empty cell - this must be covered by some piece
    first_empty = _find_first_empty_cell(grid, rows, cols)
    
    if first_empty is None:
        # No empty cells but still have pieces to place
        return
    
    empty_row, empty_col = first_empty
    
    # Try each remaining piece
    for i in range(piece_index, len(piece_rotations)):
        rotations = piece_rotations[i]
        
        # Get the next available number for this piece
        piece_number = _get_next_number(existing_numbers)
        
        # Try each rotation of this piece
        for rotation in rotations:
            piece_cells = _get_piece_cells(rotation)
            
            if not piece_cells:
                continue
            
            # For each cell in the piece that could cover the first empty cell
            for anchor_dr, anchor_dc in piece_cells:
                # Calculate where the piece would start if this cell covers first_empty
                start_row = empty_row - anchor_dr
                start_col = empty_col - anchor_dc
                
                # Check if piece can be placed here
                if _can_place_piece(grid, rotation, start_row, start_col, rows, cols):
                    # Place the piece
                    _place_piece(grid, rotation, start_row, start_col, piece_number)
                    existing_numbers.add(piece_number)
                    
                    # Swap current piece with piece at piece_index
                    piece_rotations[piece_index], piece_rotations[i] = \
                        piece_rotations[i], piece_rotations[piece_index]
                    
                    # Recurse
                    _backtrack(grid, piece_rotations, piece_index + 1, 
                              existing_numbers, solution_count, rows, cols)
                    
                    # Swap back
                    piece_rotations[piece_index], piece_rotations[i] = \
                        piece_rotations[i], piece_rotations[piece_index]
                    
                    # Remove the piece
                    _remove_piece(grid, rotation, start_row, start_col)
                    existing_numbers.remove(piece_number)