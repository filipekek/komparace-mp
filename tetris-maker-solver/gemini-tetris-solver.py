def solve_puzzle(grid, pieces):
    """
    Solves a Tetris-style puzzle by fitting pieces into a grid.
    
    Args:
        grid (list[list[int]]): A square grid where 0 is empty and non-zero is blocked.
        pieces (list[list[list[int]]]): A list of pieces (0/1 matrices).
        
    Returns:
        int: The number of distinct solutions found.
    """
    
    # --- Helper Functions ---

    def get_matrix_dimensions(matrix):
        """Returns height and width of a matrix."""
        if not matrix:
            return 0, 0
        return len(matrix), len(matrix[0])

    def trim_piece(piece):
        """Removes empty rows and columns from a piece definition."""
        if not piece:
            return []
            
        # Find rows with at least one 1
        non_empty_rows = [i for i, row in enumerate(piece) if any(row)]
        if not non_empty_rows:
            return []
            
        r_min, r_max = non_empty_rows[0], non_empty_rows[-1]
        
        # Find cols with at least one 1
        non_empty_cols = []
        rows = piece[r_min : r_max + 1]
        num_cols = len(rows[0])
        
        for c in range(num_cols):
            if any(row[c] for row in rows):
                non_empty_cols.append(c)
                
        c_min, c_max = non_empty_cols[0], non_empty_cols[-1]
        
        trimmed = []
        for r in range(r_min, r_max + 1):
            trimmed.append(piece[r][c_min : c_max + 1])
            
        return trimmed

    def rotate_90(matrix):
        """Rotates a matrix 90 degrees clockwise."""
        if not matrix:
            return []
        h, w = get_matrix_dimensions(matrix)
        new_matrix = [[0] * h for _ in range(w)]
        for r in range(h):
            for c in range(w):
                new_matrix[c][h - 1 - r] = matrix[r][c]
        return new_matrix

    def flip_horizontal(matrix):
        """Flips a matrix horizontally."""
        return [row[::-1] for row in matrix]

    def generate_orientations(piece):
        """Generates all unique rotations and flips of a piece."""
        piece = trim_piece(piece)
        if not piece:
            return []
            
        variations = []
        current = piece
        
        # We perform 4 rotations
        for _ in range(4):
            variations.append(current)
            # Also add the flipped version of this rotation
            variations.append(flip_horizontal(current))
            current = rotate_90(current)
            
        # Deduplicate variations based on tuple representation
        unique_variations = []
        seen = set()
        
        for var in variations:
            # Convert to tuple of tuples for hashing
            t_var = tuple(tuple(row) for row in var)
            if t_var not in seen:
                seen.add(t_var)
                unique_variations.append(var)
                
        return unique_variations

    def count_piece_area(piece):
        """Counts the number of 1s in a piece."""
        return sum(sum(row) for row in piece)

    def can_place(grid, piece, r_offset, c_offset):
        """Checks if a piece can be placed at grid[r_offset][c_offset]."""
        p_h, p_w = get_matrix_dimensions(piece)
        g_h, g_w = get_matrix_dimensions(grid)
        
        # Check bounds
        if r_offset + p_h > g_h or c_offset + p_w > g_w:
            return False
            
        # Check overlap
        for r in range(p_h):
            for c in range(p_w):
                if piece[r][c] == 1:
                    if grid[r_offset + r][c_offset + c] != 0:
                        return False
        return True

    def place_piece(grid, piece, r_offset, c_offset, piece_id):
        """Places a piece on the grid (in-place)."""
        p_h, p_w = get_matrix_dimensions(piece)
        for r in range(p_h):
            for c in range(p_w):
                if piece[r][c] == 1:
                    grid[r_offset + r][c_offset + c] = piece_id

    def remove_piece(grid, piece, r_offset, c_offset):
        """Removes a piece from the grid (in-place)."""
        p_h, p_w = get_matrix_dimensions(piece)
        for r in range(p_h):
            for c in range(p_w):
                if piece[r][c] == 1:
                    grid[r_offset + r][c_offset + c] = 0

    # --- Main Logic ---

    # 1. Validate Input and Basic Constraints
    grid_rows = len(grid)
    if grid_rows == 0:
        return 0
    grid_cols = len(grid[0])
    
    # Calculate empty space in grid
    existing_obstacles = set()
    empty_cells = 0
    for r in range(grid_rows):
        for c in range(grid_cols):
            val = grid[r][c]
            if val == 0:
                empty_cells += 1
            else:
                existing_obstacles.add(val)
                
    # Calculate total area required by pieces
    pieces_area = 0
    for p in pieces:
        pieces_area += count_piece_area(p)
        
    # Correctness check: Pieces must exactly fill the empty space
    if pieces_area != empty_cells:
        return 0

    # 2. Assign unique IDs to new pieces
    # We find IDs starting from 1 that are not in existing_obstacles
    piece_ids = []
    current_id = 1
    for _ in range(len(pieces)):
        while current_id in existing_obstacles:
            current_id += 1
        piece_ids.append(current_id)
        current_id += 1

    # 3. Pre-calculate all unique orientations for every piece
    # This stores a list of lists: [[shape1_v1, shape1_v2...], [shape2_v1...], ...]
    all_piece_variations = [generate_orientations(p) for p in pieces]

    # 4. Backtracking Solver
    solution_count = 0

    def backtrack(piece_idx):
        nonlocal solution_count
        
        # Base Case: All pieces placed
        if piece_idx == len(pieces):
            solution_count += 1
            return

        current_variations = all_piece_variations[piece_idx]
        p_id = piece_ids[piece_idx]
        
        # Attempt to place current piece in all orientations
        for shape in current_variations:
            h, w = get_matrix_dimensions(shape)
            
            # Iterate through all possible top-left positions
            # Optimization: The search range is restricted by piece dimensions
            for r in range(grid_rows - h + 1):
                for c in range(grid_cols - w + 1):
                    
                    if can_place(grid, shape, r, c):
                        # Apply move
                        place_piece(grid, shape, r, c, p_id)
                        
                        # Recurse
                        backtrack(piece_idx + 1)
                        
                        # Backtrack (Undo move)
                        remove_piece(grid, shape, r, c)

    # Start the solver
    backtrack(0)
    
    return solution_count