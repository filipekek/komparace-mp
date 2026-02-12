def solve_puzzle(grid, pieces):
    """
    Solves a Tetris-style puzzle by placing pieces into a grid.
    
    Args:
        grid: A 2D list representing the puzzle grid (list of lists).
              0 represents empty cells, non-zero values are fixed pieces.
        pieces: A list of 2D lists, where each piece is represented by 0s and 1s.
                1s indicate the shape of the piece.
    
    Returns:
        The number of solutions found.
    """
    # Input validation
    if not grid or not all(grid):
        return 0
    if not pieces:
        return 0
    
    # Validate grid is rectangular
    rows = len(grid)
    cols = len(grid[0])
    if not all(len(row) == cols for row in grid):
        raise ValueError("Grid must be rectangular")
    
    # Create a working copy of the grid
    work_grid = [row[:] for row in grid]
    
    # Find the next available piece number
    max_num = 0
    for row in work_grid:
        for cell in row:
            if cell > max_num:
                max_num = cell
    
    # Generate all orientations for each piece
    all_piece_orientations = []
    for piece in pieces:
        orientations = generate_orientations(piece)
        all_piece_orientations.append(orientations)
    
    # Track which pieces have been used
    used_pieces = [False] * len(pieces)
    
    # Start solving
    solution_count = [0]  # Use list to allow modification in nested function
    backtrack(work_grid, all_piece_orientations, used_pieces, max_num + 1, solution_count)
    
    return solution_count[0]


def generate_orientations(piece):
    """
    Generate all unique orientations of a piece (rotations and flips).
    
    Args:
        piece: A 2D list representing a piece with 0s and 1s.
    
    Returns:
        A list of unique orientations of the piece.
    """
    if not piece or not piece[0]:
        return []
    
    orientations = []
    seen = set()
    
    # Generate rotations and flips
    current = piece
    for _ in range(4):  # 4 rotations
        # Add current rotation
        normalized = normalize_piece(current)
        signature = piece_to_tuple(normalized)
        if signature not in seen:
            seen.add(signature)
            orientations.append(normalized)
        
        # Add flipped version
        flipped = flip_piece(current)
        normalized_flip = normalize_piece(flipped)
        signature_flip = piece_to_tuple(normalized_flip)
        if signature_flip not in seen:
            seen.add(signature_flip)
            orientations.append(normalized_flip)
        
        # Rotate for next iteration
        current = rotate_piece(current)
    
    return orientations


def rotate_piece(piece):
    """
    Rotate a piece 90 degrees clockwise.
    
    Args:
        piece: A 2D list representing a piece.
    
    Returns:
        The rotated piece.
    """
    if not piece or not piece[0]:
        return piece
    
    rows = len(piece)
    cols = len(piece[0])
    rotated = [[0] * rows for _ in range(cols)]
    
    for i in range(rows):
        for j in range(cols):
            rotated[j][rows - 1 - i] = piece[i][j]
    
    return rotated


def flip_piece(piece):
    """
    Flip a piece horizontally (mirror transformation).
    
    Args:
        piece: A 2D list representing a piece.
    
    Returns:
        The flipped piece.
    """
    if not piece:
        return piece
    
    return [row[::-1] for row in piece]


def normalize_piece(piece):
    """
    Normalize a piece by removing empty rows and columns from edges.
    
    Args:
        piece: A 2D list representing a piece.
    
    Returns:
        The normalized piece.
    """
    if not piece or not piece[0]:
        return piece
    
    # Remove empty rows from top and bottom
    while piece and all(cell == 0 for cell in piece[0]):
        piece = piece[1:]
    
    while piece and all(cell == 0 for cell in piece[-1]):
        piece = piece[:-1]
    
    if not piece:
        return []
    
    # Remove empty columns from left and right
    cols = len(piece[0])
    
    # Find first non-empty column
    left = 0
    while left < cols and all(row[left] == 0 for row in piece):
        left += 1
    
    # Find last non-empty column
    right = cols - 1
    while right >= 0 and all(row[right] == 0 for row in piece):
        right -= 1
    
    if left > right:
        return []
    
    # Extract non-empty columns
    normalized = [row[left:right+1] for row in piece]
    
    return normalized


def piece_to_tuple(piece):
    """
    Convert a piece to a tuple for hashing and comparison.
    
    Args:
        piece: A 2D list representing a piece.
    
    Returns:
        A tuple representation of the piece.
    """
    return tuple(tuple(row) for row in piece)


def backtrack(grid, all_piece_orientations, used_pieces, next_piece_num, solution_count):
    """
    Backtracking algorithm to find all solutions.
    
    Args:
        grid: Current state of the grid.
        all_piece_orientations: List of all orientations for each piece.
        used_pieces: Boolean list tracking which pieces have been used.
        next_piece_num: The next number to assign to a placed piece.
        solution_count: List containing the count of solutions found.
    """
    # Find the next empty cell
    empty_cell = find_next_empty_cell(grid)
    
    if empty_cell is None:
        # No empty cells left - check if all pieces are used
        if all(used_pieces):
            solution_count[0] += 1
        return
    
    row, col = empty_cell
    
    # Try each unused piece
    for piece_idx in range(len(all_piece_orientations)):
        if used_pieces[piece_idx]:
            continue
        
        # Try each orientation of the piece
        for orientation in all_piece_orientations[piece_idx]:
            if can_place_piece(grid, orientation, row, col):
                # Place the piece
                place_piece(grid, orientation, row, col, next_piece_num)
                used_pieces[piece_idx] = True
                
                # Recurse
                backtrack(grid, all_piece_orientations, used_pieces, next_piece_num + 1, solution_count)
                
                # Backtrack - remove the piece
                remove_piece(grid, orientation, row, col)
                used_pieces[piece_idx] = False
    
    # If no piece fits at this position, return (dead end)
    return


def find_next_empty_cell(grid):
    """
    Find the next empty cell (with value 0) in the grid.
    Uses row-major order for consistency.
    
    Args:
        grid: The current grid state.
    
    Returns:
        A tuple (row, col) of the next empty cell, or None if no empty cells exist.
    """
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 0:
                return (i, j)
    return None


def can_place_piece(grid, piece, start_row, start_col):
    """
    Check if a piece can be placed at the given position.
    
    Args:
        grid: The current grid state.
        piece: The piece to place.
        start_row: Starting row position.
        start_col: Starting column position.
    
    Returns:
        True if the piece can be placed, False otherwise.
    """
    if not piece or not piece[0]:
        return False
    
    rows = len(grid)
    cols = len(grid[0])
    piece_rows = len(piece)
    piece_cols = len(piece[0])
    
    # Check if piece fits within grid bounds
    if start_row + piece_rows > rows or start_col + piece_cols > cols:
        return False
    
    # Check if piece overlaps with non-empty cells
    for i in range(piece_rows):
        for j in range(piece_cols):
            if piece[i][j] == 1:  # Part of the piece
                if grid[start_row + i][start_col + j] != 0:  # Cell is not empty
                    return False
    
    return True


def place_piece(grid, piece, start_row, start_col, piece_num):
    """
    Place a piece on the grid.
    
    Args:
        grid: The current grid state.
        piece: The piece to place.
        start_row: Starting row position.
        start_col: Starting column position.
        piece_num: The number to assign to this piece.
    """
    piece_rows = len(piece)
    piece_cols = len(piece[0])
    
    for i in range(piece_rows):
        for j in range(piece_cols):
            if piece[i][j] == 1:
                grid[start_row + i][start_col + j] = piece_num


def remove_piece(grid, piece, start_row, start_col):
    """
    Remove a piece from the grid (set cells back to 0).
    
    Args:
        grid: The current grid state.
        piece: The piece to remove.
        start_row: Starting row position.
        start_col: Starting column position.
    """
    piece_rows = len(piece)
    piece_cols = len(piece[0])
    
    for i in range(piece_rows):
        for j in range(piece_cols):
            if piece[i][j] == 1:
                grid[start_row + i][start_col + j] = 0