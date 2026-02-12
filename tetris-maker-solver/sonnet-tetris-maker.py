import random
import time

def create_tetris_puzzle(grid_size=10, piece_min_size=3, piece_max_size=7, clue_percentage=0.3):
    """
    Creates a random Tetris-styled puzzle.
    
    Args:
        grid_size: Size of the square grid (default 10x10)
        piece_min_size: Minimum number of cells in a piece (default 3)
        piece_max_size: Maximum number of cells in a piece (default 7)
        clue_percentage: Percentage of pieces to keep as clues (default 0.3)
    
    Returns:
        tuple: (empty_grid, pieces, full_grid)
            - empty_grid: Grid with only clue pieces
            - pieces: List of piece masks (0s and 1s)
            - full_grid: Complete solution grid
    """
    
    # Initialize empty grid
    full_grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
    
    # Track which cells are occupied
    occupied = [[False for _ in range(grid_size)] for _ in range(grid_size)]
    
    piece_id = 1
    piece_positions = []  # Store piece info: (piece_id, cells)
    
    # Fill the grid with pieces
    max_attempts = 1000
    attempt = 0
    
    while attempt < max_attempts:
        # Check if grid is full enough (e.g., 85% filled)
        total_cells = grid_size * grid_size
        filled_cells = sum(sum(row) for row in occupied)
        
        if filled_cells >= total_cells * 0.85:
            break
        
        # Generate a random piece
        piece_size = random.randint(piece_min_size, piece_max_size)
        piece_cells = generate_random_piece(piece_size)
        
        # Try to place the piece
        placed = False
        placement_attempts = 0
        max_placement_attempts = 100
        
        while not placed and placement_attempts < max_placement_attempts:
            start_row = random.randint(0, grid_size - 1)
            start_col = random.randint(0, grid_size - 1)
            
            if try_place_piece(full_grid, occupied, piece_cells, start_row, start_col, 
                             piece_id, grid_size):
                # Successfully placed
                absolute_cells = [(start_row + dr, start_col + dc) for dr, dc in piece_cells]
                piece_positions.append((piece_id, absolute_cells))
                piece_id += 1
                placed = True
            
            placement_attempts += 1
        
        if not placed:
            attempt += 1
        else:
            attempt = 0  # Reset attempt counter on success
    
    # If we couldn't fill enough, fill remaining spaces with small pieces
    fill_remaining_cells(full_grid, occupied, piece_id, grid_size)
    
    # Separate pieces into clues and removable pieces
    random.shuffle(piece_positions)
    num_clues = max(1, int(len(piece_positions) * clue_percentage))
    clue_pieces = piece_positions[:num_clues]
    removable_pieces = piece_positions[num_clues:]
    
    # Create empty grid with only clue pieces
    empty_grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
    for piece_id, cells in clue_pieces:
        for row, col in cells:
            empty_grid[row][col] = piece_id
    
    # Extract removable pieces as 0/1 masks
    pieces = []
    for piece_id, cells in removable_pieces:
        piece_mask = extract_piece_mask(cells)
        pieces.append(piece_mask)
    
    # Shuffle the pieces
    random.shuffle(pieces)
    
    return empty_grid, pieces, full_grid


def generate_random_piece(size):
    """
    Generates a random connected piece of given size.
    Returns list of (row, col) relative coordinates.
    """
    if size <= 0:
        return []
    
    # Start with initial cell at (0, 0)
    cells = [(0, 0)]
    
    # Add cells one by one, ensuring connectivity
    for _ in range(size - 1):
        # Find all possible adjacent positions
        candidates = []
        for row, col in cells:
            # Check all 4 directions
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                new_pos = (row + dr, col + dc)
                if new_pos not in cells:
                    candidates.append(new_pos)
        
        if not candidates:
            break
        
        # Add a random candidate
        new_cell = random.choice(candidates)
        cells.append(new_cell)
    
    # Normalize to start from (0, 0)
    min_row = min(r for r, c in cells)
    min_col = min(c for r, c in cells)
    normalized = [(r - min_row, c - min_col) for r, c in cells]
    
    return normalized


def try_place_piece(grid, occupied, piece_cells, start_row, start_col, piece_id, grid_size):
    """
    Attempts to place a piece on the grid.
    Returns True if successful, False otherwise.
    """
    # Check if all cells can be placed
    cells_to_place = []
    for dr, dc in piece_cells:
        row = start_row + dr
        col = start_col + dc
        
        # Check bounds
        if row < 0 or row >= grid_size or col < 0 or col >= grid_size:
            return False
        
        # Check if occupied
        if occupied[row][col]:
            return False
        
        cells_to_place.append((row, col))
    
    # Place the piece
    for row, col in cells_to_place:
        grid[row][col] = piece_id
        occupied[row][col] = True
    
    return True


def fill_remaining_cells(grid, occupied, start_piece_id, grid_size):
    """
    Fills any remaining empty cells with small pieces.
    """
    piece_id = start_piece_id
    
    for row in range(grid_size):
        for col in range(grid_size):
            if not occupied[row][col]:
                # Try to create a small connected piece from this cell
                cells = [(row, col)]
                occupied[row][col] = True
                grid[row][col] = piece_id
                
                # Try to add 1-3 more adjacent cells
                target_size = random.randint(2, 4)
                for _ in range(target_size - 1):
                    candidates = []
                    for r, c in cells:
                        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                            nr, nc = r + dr, c + dc
                            if (0 <= nr < grid_size and 0 <= nc < grid_size and 
                                not occupied[nr][nc]):
                                candidates.append((nr, nc))
                    
                    if candidates:
                        new_cell = random.choice(candidates)
                        cells.append(new_cell)
                        occupied[new_cell[0]][new_cell[1]] = True
                        grid[new_cell[0]][new_cell[1]] = piece_id
                    else:
                        break
                
                piece_id += 1


def extract_piece_mask(cells):
    """
    Extracts a piece as a 0/1 mask (bounding box representation).
    """
    if not cells:
        return [[]]
    
    rows = [r for r, c in cells]
    cols = [c for r, c in cells]
    
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)
    
    height = max_row - min_row + 1
    width = max_col - min_col + 1
    
    # Create mask
    mask = [[0 for _ in range(width)] for _ in range(height)]
    
    for row, col in cells:
        mask[row - min_row][col - min_col] = 1
    
    return mask


def print_grid(grid, title="Grid"):
    """
    Prints a grid in a readable format.
    """
    print(f"\n{title}:")
    print("-" * (len(grid[0]) * 3 + 1))
    for row in grid:
        print("|" + "|".join(f"{cell:2}" for cell in row) + "|")
    print("-" * (len(grid[0]) * 3 + 1))


def print_piece(piece, index):
    """
    Prints a piece mask.
    """
    print(f"\nPiece {index + 1}:")
    for row in piece:
        print(" ".join(str(cell) for cell in row))


def main():
    """
    Main function to demonstrate the puzzle generator.
    """
    print("Generating Tetris Puzzle...")
    
    # Generate puzzle
    empty_grid, pieces, full_grid = create_tetris_puzzle(
        grid_size=10,
        piece_min_size=3,
        piece_max_size=7,
        clue_percentage=0.3
    )
    
    # Display results
    print_grid(empty_grid, "Empty Grid (with clues)")
    
    print(f"\n\nNumber of pieces to place: {len(pieces)}")
    for i, piece in enumerate(pieces):
        print_piece(piece, i)
    
    print_grid(full_grid, "Solution (Full Grid)")
    
    # Verify grid is fully filled
    total_cells = len(full_grid) * len(full_grid[0])
    filled_cells = sum(1 for row in full_grid for cell in row if cell > 0)
    print(f"\nGrid filled: {filled_cells}/{total_cells} cells ({filled_cells/total_cells*100:.1f}%)")
    
    # Return the variables as specified
    return empty_grid, pieces, full_grid


if __name__ == "__main__":
    t1 = time.perf_counter()
    empty_grid, pieces, full_grid = main()
    t2 = time.perf_counter()
    print(t2-t1)