import random
import copy
import time

def generate_tetris_puzzle(rows=10, cols=10, min_size=3, max_size=7, clue_chance=0.3):
    """
    Generates a Tetris-styled puzzle.
    
    Args:
        rows (int): Grid height.
        cols (int): Grid width.
        min_size (int): Min blocks per piece.
        max_size (int): Max blocks per piece.
        clue_chance (float): Probability (0-1) of a piece remaining in the grid as a clue.
        
    Returns:
        tuple: (puzzle_grid, pieces_list, solution_grid)
    """

    # --- Helper: Get neighbors ---
    def get_neighbors(r, c, grid_h, grid_w):
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        result = []
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid_h and 0 <= nc < grid_w:
                result.append((nr, nc))
        return result

    # --- Step 1: Generate the Solution Grid ---
    # We loop until we get a valid generation (sometimes random fills create unfixable grids)
    solution_grid = []
    
    valid_generation = False
    while not valid_generation:
        # Initialize 0 grid
        temp_grid = [[0 for _ in range(cols)] for _ in range(rows)]
        
        # 1. Fill with random shapes
        current_id = 1
        free_cells = [(r, c) for r in range(rows) for c in range(cols)]
        random.shuffle(free_cells)
        
        visited = set()

        for start_node in free_cells:
            if start_node in visited:
                continue
            
            # Start a new shape
            target_size = random.randint(min_size, max_size)
            shape_coords = [start_node]
            visited.add(start_node)
            temp_grid[start_node[0]][start_node[1]] = current_id
            
            # Grow shape using randomized BFS
            queue = [start_node]
            
            while len(shape_coords) < target_size and queue:
                # Pick random element from queue to make shapes irregular
                curr = queue.pop(random.randint(0, len(queue)-1)) 
                
                neighbors = get_neighbors(curr[0], curr[1], rows, cols)
                random.shuffle(neighbors)
                
                for nr, nc in neighbors:
                    if (nr, nc) not in visited and len(shape_coords) < target_size:
                        temp_grid[nr][nc] = current_id
                        visited.add((nr, nc))
                        shape_coords.append((nr, nc))
                        queue.append((nr, nc))
            
            current_id += 1

        # 2. Repair Phase: Merge small pieces into neighbors
        # Sometimes we are left with size 1 or 2 pieces. We must merge them.
        changed = True
        while changed:
            changed = False
            # Group coordinates by ID
            shapes = {}
            for r in range(rows):
                for c in range(cols):
                    pid = temp_grid[r][c]
                    if pid not in shapes: shapes[pid] = []
                    shapes[pid].append((r, c))
            
            # check sizes
            for pid, coords in shapes.items():
                if len(coords) < min_size:
                    # This piece is too small. Find a neighbor piece to merge into.
                    neighbor_ids = set()
                    for r, c in coords:
                        for nr, nc in get_neighbors(r, c, rows, cols):
                            nid = temp_grid[nr][nc]
                            if nid != pid:
                                neighbor_ids.add(nid)
                    
                    if neighbor_ids:
                        # Merge into the first valid neighbor found
                        target_id = list(neighbor_ids)[0]
                        for r, c in coords:
                            temp_grid[r][c] = target_id
                        changed = True 
                        break # Restart scan after a merge to ensure stability

        # Final Validation check
        # Ensure no piece is < min_size (Merge logic usually fixes this, but we double check)
        shapes = {}
        for r in range(rows):
            for c in range(cols):
                pid = temp_grid[r][c]
                if pid not in shapes: shapes[pid] = []
                shapes[pid].append((r, c))
        
        if all(len(s) >= min_size for s in shapes.values()):
            valid_generation = True
            solution_grid = temp_grid

    # --- Step 2: Separate Clues and Pieces ---
    
    # Identify unique pieces and their coordinates
    piece_map = {}
    for r in range(rows):
        for c in range(cols):
            pid = solution_grid[r][c]
            if pid not in piece_map:
                piece_map[pid] = []
            piece_map[pid].append((r, c))

    puzzle_grid = [row[:] for row in solution_grid] # Deep copy
    pieces_output = []

    # Get list of all IDs
    all_ids = list(piece_map.keys())
    
    for pid in all_ids:
        # Decide if this is a clue or a piece to be removed
        is_clue = random.random() < clue_chance
        
        if is_clue:
            # Keep in grid, do nothing (it's already in puzzle_grid)
            pass
        else:
            # Remove from puzzle grid
            coords = piece_map[pid]
            for r, c in coords:
                puzzle_grid[r][c] = 0
            
            # Create the 0/1 matrix for the piece
            # 1. Find bounding box
            rs = [x[0] for x in coords]
            cs = [x[1] for x in coords]
            min_r, max_r = min(rs), max(rs)
            min_c, max_c = min(cs), max(cs)
            
            h = max_r - min_r + 1
            w = max_c - min_c + 1
            
            # 2. Fill matrix
            piece_matrix = [[0 for _ in range(w)] for _ in range(h)]
            for r, c in coords:
                # Shift coordinates relative to bounding box
                piece_matrix[r - min_r][c - min_c] = 1
                
            pieces_output.append(piece_matrix)

    # Shuffle the extracted pieces so the order doesn't give away the position
    random.shuffle(pieces_output)

    return puzzle_grid, pieces_output, solution_grid

# --- Execution and Display ---

def print_grid(name, grid):
    print(f"--- {name} ---")
    if not grid:
        print("Empty")
        return
    for row in grid:
        # Formatting for nicer alignment
        print([f"{x:2}" for x in row])
    print()

def print_pieces(pieces):
    print(f"--- Extracted Pieces ({len(pieces)}) ---")
    for i, p in enumerate(pieces):
        print(f"Piece {i+1}:")
        for row in p:
            print(row)
        print()

if __name__ == "__main__":
    # Generate the puzzle
    t1 = time.perf_counter()
    try:
        empty_grid, pieces, full_grid = generate_tetris_puzzle()
        
        # Display results
        print_grid("Solution Grid (Answer)", full_grid)
        print_grid("Puzzle Grid (With Clues)", empty_grid)
        print_pieces(pieces)
        
        # Verification
        print("Verification:")
        print(f"Grid Size: {len(full_grid)}x{len(full_grid[0])}")
        print(f"Number of pieces to place: {len(pieces)}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
    
    t2 = time.perf_counter()
    print(t2-t1)