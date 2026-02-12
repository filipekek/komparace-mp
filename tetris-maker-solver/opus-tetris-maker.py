"""
Tetris-Style Puzzle Generator

This module generates random Tetris-style puzzles on a 10x10 grid.
Pieces are connected shapes (3-7 elements) that fill the grid.
Some pieces remain as clues, others are extracted for the player to place.
"""

import random
from copy import deepcopy
import time


def get_neighbors(row, col, rows, cols):
    """Get valid neighboring positions (up, down, left, right)."""
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = row + dr, col + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            neighbors.append((nr, nc))
    return neighbors


def is_connected(cells):
    """Check if a set of cells forms a connected region using BFS."""
    if not cells:
        return True
    if len(cells) == 1:
        return True
    
    cells_set = set(cells)
    visited = set()
    start = next(iter(cells_set))
    queue = [start]
    visited.add(start)
    
    while queue:
        r, c = queue.pop(0)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) in cells_set and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc))
    
    return len(visited) == len(cells_set)


def count_empty_cells(grid):
    """Count the number of empty cells in the grid."""
    return sum(1 for row in grid for cell in row if cell == 0)


def get_empty_neighbors_count(grid, row, col):
    """Count empty neighbors for a cell."""
    rows, cols = len(grid), len(grid[0])
    count = 0
    for nr, nc in get_neighbors(row, col, rows, cols):
        if grid[nr][nc] == 0:
            count += 1
    return count


def generate_piece(grid, start_row, start_col, piece_id, min_size=3, max_size=7):
    """
    Generate a single connected piece starting from a given position.
    Uses a randomized growth algorithm to create organic Tetris-like shapes.
    
    Args:
        grid: The current grid state
        start_row, start_col: Starting position for the piece
        piece_id: Unique identifier for this piece
        min_size: Minimum number of cells in the piece (default 3)
        max_size: Maximum number of cells in the piece (default 7)
    
    Returns:
        List of (row, col) positions that make up the piece, or None if failed
    """
    rows, cols = len(grid), len(grid[0])
    target_size = random.randint(min_size, max_size)
    
    piece_cells = [(start_row, start_col)]
    frontier = list(get_neighbors(start_row, start_col, rows, cols))
    random.shuffle(frontier)
    
    while len(piece_cells) < target_size and frontier:
        # Pick a random cell from frontier
        idx = random.randint(0, len(frontier) - 1)
        candidate = frontier.pop(idx)
        cr, cc = candidate
        
        # Check if cell is empty
        if grid[cr][cc] != 0:
            continue
        
        # Check if cell is adjacent to existing piece
        is_adjacent = False
        for pr, pc in piece_cells:
            if abs(pr - cr) + abs(pc - cc) == 1:
                is_adjacent = True
                break
        
        if not is_adjacent:
            continue
        
        # Add cell to piece
        piece_cells.append((cr, cc))
        
        # Add new neighbors to frontier
        for nr, nc in get_neighbors(cr, cc, rows, cols):
            if grid[nr][nc] == 0 and (nr, nc) not in piece_cells and (nr, nc) not in frontier:
                frontier.append((nr, nc))
        
        random.shuffle(frontier)
    
    # Only return piece if it meets minimum size requirement
    if len(piece_cells) >= min_size:
        return piece_cells
    return None


def split_piece(grid, piece_map, piece_id, max_size, min_size):
    """
    Split a piece that is larger than max_size into smaller valid pieces.
    
    Returns True if split was successful, False otherwise.
    """
    rows, cols = len(grid), len(grid[0])
    cells = list(piece_map[piece_id])
    
    if len(cells) <= max_size:
        return True
    
    # We need to split this piece
    # Try to carve out pieces of size min_size to max_size
    while len(cells) > max_size:
        # Find edge cells (cells with fewer piece neighbors)
        edge_cells = []
        for r, c in cells:
            piece_neighbor_count = sum(1 for nr, nc in get_neighbors(r, c, rows, cols) 
                                       if (nr, nc) in cells)
            if piece_neighbor_count <= 2:
                edge_cells.append((r, c))
        
        if not edge_cells:
            edge_cells = cells[:1]
        
        # Start a new piece from an edge cell
        start_cell = random.choice(edge_cells)
        new_piece_cells = [start_cell]
        remaining = [c for c in cells if c != start_cell]
        
        # Grow the new piece
        target_new_size = random.randint(min_size, min(max_size, len(cells) - min_size))
        
        while len(new_piece_cells) < target_new_size:
            # Find cells adjacent to new_piece that are in remaining
            candidates = []
            for r, c in new_piece_cells:
                for nr, nc in get_neighbors(r, c, rows, cols):
                    if (nr, nc) in remaining and (nr, nc) not in new_piece_cells:
                        candidates.append((nr, nc))
            
            if not candidates:
                break
            
            # Pick candidate that maintains connectivity of remaining
            valid_candidate = None
            random.shuffle(candidates)
            for candidate in candidates:
                test_remaining = [c for c in remaining if c != candidate]
                if is_connected(test_remaining):
                    valid_candidate = candidate
                    break
            
            if valid_candidate is None:
                break
            
            new_piece_cells.append(valid_candidate)
            remaining = [c for c in remaining if c != valid_candidate]
        
        # Check if split is valid
        if len(new_piece_cells) >= min_size and len(remaining) >= min_size and is_connected(remaining):
            # Create new piece
            new_pid = max(piece_map.keys()) + 1
            for r, c in new_piece_cells:
                grid[r][c] = new_pid
            piece_map[new_pid] = new_piece_cells
            
            # Update old piece
            piece_map[piece_id] = remaining
            cells = remaining
        else:
            # Can't split properly, give up
            return False
    
    return True


def fill_grid_with_pieces(rows=10, cols=10, min_piece_size=3, max_piece_size=7):
    """
    Fill a grid with connected pieces, each having 3-7 cells.
    
    Args:
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        min_piece_size: Minimum size for each piece
        max_piece_size: Maximum size for each piece
    
    Returns:
        Tuple of (filled_grid, piece_map) where piece_map maps piece_id to cell positions
    """
    max_global_attempts = 100
    
    for global_attempt in range(max_global_attempts):
        grid = [[0] * cols for _ in range(rows)]
        piece_map = {}
        piece_id = 1
        
        max_attempts = 1000
        attempts = 0
        
        while count_empty_cells(grid) > 0 and attempts < max_attempts:
            attempts += 1
            
            # Find empty cells, prioritize constrained cells
            empty_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 0]
            
            if not empty_cells:
                break
            
            # Sort by number of empty neighbors (ascending) to avoid isolated pockets
            empty_cells.sort(key=lambda cell: get_empty_neighbors_count(grid, cell[0], cell[1]))
            
            # Pick from constrained cells with some randomness
            if len(empty_cells) > 5:
                constrained = empty_cells[:max(3, len(empty_cells) // 4)]
                start_cell = random.choice(constrained)
            else:
                start_cell = empty_cells[0]
            
            # Adjust size based on remaining space
            remaining = count_empty_cells(grid)
            effective_max = min(max_piece_size, remaining)
            effective_min = min(min_piece_size, remaining)
            
            if effective_min > effective_max:
                effective_min = effective_max
            
            # Generate piece
            piece_cells = generate_piece(
                grid, start_cell[0], start_cell[1], piece_id,
                effective_min, effective_max
            )
            
            if piece_cells:
                for r, c in piece_cells:
                    grid[r][c] = piece_id
                piece_map[piece_id] = list(piece_cells)
                piece_id += 1
        
        # Handle remaining cells by extending adjacent pieces
        remaining_cells = [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 0]
        
        for r, c in remaining_cells:
            candidates = []
            for nr, nc in get_neighbors(r, c, rows, cols):
                if grid[nr][nc] != 0:
                    adj_pid = grid[nr][nc]
                    if len(piece_map[adj_pid]) < max_piece_size:
                        candidates.append(adj_pid)
            
            if candidates:
                target_pid = min(candidates, key=lambda p: len(piece_map[p]))
                grid[r][c] = target_pid
                piece_map[target_pid].append((r, c))
            else:
                # Extend any adjacent piece
                for nr, nc in get_neighbors(r, c, rows, cols):
                    if grid[nr][nc] != 0:
                        adj_pid = grid[nr][nc]
                        grid[r][c] = adj_pid
                        piece_map[adj_pid].append((r, c))
                        break
        
        # Split any pieces that are too large
        pieces_to_split = [pid for pid, cells in piece_map.items() if len(cells) > max_piece_size]
        for pid in pieces_to_split:
            split_piece(grid, piece_map, pid, max_piece_size, min_piece_size)
        
        # Validate: all pieces must be 3-7 cells, grid must be full
        all_valid = True
        
        # Check for unfilled cells
        if count_empty_cells(grid) > 0:
            all_valid = False
        
        # Check piece sizes
        for pid, cells in piece_map.items():
            if len(cells) < min_piece_size or len(cells) > max_piece_size:
                all_valid = False
                break
            if not is_connected(cells):
                all_valid = False
                break
        
        if all_valid:
            return grid, piece_map
    
    # Return best attempt (may not be perfect)
    return grid, piece_map


def extract_piece_shape(piece_cells):
    """
    Extract a piece as a minimal bounding box with 0s and 1s.
    
    Args:
        piece_cells: List of (row, col) positions for this piece
    
    Returns:
        A list of lists representing the piece shape (0s and 1s)
    """
    if not piece_cells:
        return [[]]
    
    min_row = min(r for r, c in piece_cells)
    max_row = max(r for r, c in piece_cells)
    min_col = min(c for r, c in piece_cells)
    max_col = max(c for r, c in piece_cells)
    
    height = max_row - min_row + 1
    width = max_col - min_col + 1
    
    shape = [[0] * width for _ in range(height)]
    
    for r, c in piece_cells:
        shape[r - min_row][c - min_col] = 1
    
    return shape


def validate_solution(grid, piece_map, min_size=3, max_size=7):
    """
    Validate that the generated solution is correct.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    rows, cols = len(grid), len(grid[0])
    
    # Check all cells are filled
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 0:
                return False, f"Cell ({r}, {c}) is unfilled"
    
    # Check each piece
    for piece_id, cells in piece_map.items():
        # Check size constraints
        if len(cells) < min_size:
            return False, f"Piece {piece_id} has {len(cells)} cells (min: {min_size})"
        if len(cells) > max_size:
            return False, f"Piece {piece_id} has {len(cells)} cells (max: {max_size})"
        
        # Check connectivity
        if not is_connected(cells):
            return False, f"Piece {piece_id} is not connected"
        
        # Verify cells match grid
        for r, c in cells:
            if grid[r][c] != piece_id:
                return False, f"Cell ({r}, {c}) should be {piece_id} but is {grid[r][c]}"
    
    return True, "Solution is valid"


def generate_puzzle(clue_ratio=0.3, min_piece_size=3, max_piece_size=7, grid_size=10):
    """
    Generate a complete Tetris-style puzzle.
    
    Args:
        clue_ratio: Fraction of pieces to keep as clues (0.0 to 1.0)
        min_piece_size: Minimum size for pieces (3-7 range)
        max_piece_size: Maximum size for pieces (3-7 range)
        grid_size: Size of the grid (grid_size x grid_size)
    
    Returns:
        Tuple of (puzzle_grid, pieces_to_place, solution_grid)
        - puzzle_grid: Grid with clue pieces (numbered) and empty spaces (0)
        - pieces_to_place: List of piece shapes (0s and 1s) to place
        - solution_grid: Complete solution grid with all pieces numbered
    """
    # Validate inputs
    if min_piece_size < 1:
        min_piece_size = 1
    if max_piece_size > grid_size * grid_size:
        max_piece_size = grid_size * grid_size
    if min_piece_size > max_piece_size:
        min_piece_size = max_piece_size
    
    clue_ratio = max(0.0, min(1.0, clue_ratio))
    
    # Generate the filled grid
    solution_grid, piece_map = fill_grid_with_pieces(
        grid_size, grid_size, min_piece_size, max_piece_size
    )
    
    # Decide which pieces to keep as clues
    piece_ids = list(piece_map.keys())
    random.shuffle(piece_ids)
    
    num_clues = max(1, int(len(piece_ids) * clue_ratio))
    clue_pieces = set(piece_ids[:num_clues])
    extract_pieces = piece_ids[num_clues:]
    
    # Ensure we have at least one piece to place
    if not extract_pieces and len(piece_ids) > 1:
        extract_pieces = [piece_ids[-1]]
        clue_pieces.discard(piece_ids[-1])
    
    # Create the puzzle grid (with clues only)
    puzzle_grid = [[0] * grid_size for _ in range(grid_size)]
    for r in range(grid_size):
        for c in range(grid_size):
            if solution_grid[r][c] in clue_pieces:
                puzzle_grid[r][c] = solution_grid[r][c]
    
    # Extract pieces to place (in random order)
    pieces_to_place = []
    random.shuffle(extract_pieces)
    for pid in extract_pieces:
        shape = extract_piece_shape(piece_map[pid])
        pieces_to_place.append(shape)
    
    return puzzle_grid, pieces_to_place, solution_grid


def print_grid(grid, title="Grid"):
    """Pretty print a grid."""
    print(f"\n{title}:")
    print("-" * (len(grid[0]) * 4 + 1))
    for row in grid:
        print("|" + "|".join(f"{cell:3}" for cell in row) + "|")
    print("-" * (len(grid[0]) * 4 + 1))


def print_piece(piece, index=None):
    """Pretty print a piece shape."""
    title = f"Piece {index}" if index is not None else "Piece"
    cell_count = sum(sum(row) for row in piece)
    print(f"\n{title} ({cell_count} cells):")
    for row in piece:
        print(" ".join("█" if cell else "·" for cell in row))


def main():
    """Generate and display a sample puzzle."""
    # Uncomment the next line for reproducible results:
    # random.seed(42)
    
    print("=" * 50)
    print("TETRIS-STYLE PUZZLE GENERATOR")
    print("=" * 50)
    
    # Generate puzzle with pieces of 3-7 cells each
    puzzle_grid, pieces_to_place, solution_grid = generate_puzzle(
        clue_ratio=0.3,
        min_piece_size=3,
        max_piece_size=7,
        grid_size=10
    )
    
    # Display results
    print_grid(solution_grid, "SOLUTION (Answer)")
    print_grid(puzzle_grid, "PUZZLE (With Clues)")
    
    print("\n" + "=" * 50)
    print("PIECES TO PLACE:")
    print("=" * 50)
    for i, piece in enumerate(pieces_to_place, 1):
        print_piece(piece, i)
    
    # Count statistics
    unique_pieces = len(set(cell for row in solution_grid for cell in row))
    clue_count = len(set(cell for row in puzzle_grid for cell in row if cell != 0))
    
    print("\n" + "=" * 50)
    print("PUZZLE STATISTICS:")
    print("=" * 50)
    print(f"Total pieces in solution: {unique_pieces}")
    print(f"Clue pieces (fixed): {clue_count}")
    print(f"Pieces to place: {len(pieces_to_place)}")
    
    # Verify piece sizes
    print("\nPiece sizes in solution:")
    piece_sizes = {}
    for row in solution_grid:
        for cell in row:
            piece_sizes[cell] = piece_sizes.get(cell, 0) + 1
    for pid in sorted(piece_sizes.keys()):
        print(f"  Piece {pid}: {piece_sizes[pid]} cells")
    
    # Return the three required outputs
    return puzzle_grid, pieces_to_place, solution_grid


# Output variables (accessible after import)
empty_grid = None
pieces = None
full_grid = None

if __name__ == "__main__":
    t1 = time.perf_counter()
    empty_grid, pieces, full_grid = main()
    
    print("\n" + "=" * 50)
    print("OUTPUT VARIABLES:")
    print("=" * 50)
    print(f"\nempty_grid = {empty_grid}")
    print(f"\npieces = {pieces}")
    print(f"\nfull_grid = {full_grid}")
    t2 = time.perf_counter()
    print(t2-t1)