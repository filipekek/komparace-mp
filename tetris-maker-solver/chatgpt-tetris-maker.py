"""
Random Tetris-styled (polyomino) puzzle generator.

Creates a 10x10 grid fully tiled by connected pieces (orthogonal connectivity),
each piece having size 3..7. Then removes some pieces into a randomized list
of 0/1 matrices (tight bounding boxes), while leaving some numbered pieces
in the grid as fixed "clues".

Outputs three variables:
- empty_grid: 10x10, 0 for removed cells, piece_id (>0) for clue cells
- pieces: list of 0/1 matrices representing removed pieces (shuffled order)
- full_grid: 10x10, piece_id (>0) for every cell (solution/answer)
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Set, Tuple

Grid = List[List[int]]
Cell = Tuple[int, int]


def _make_grid(n: int, m: int, value: int = 0) -> Grid:
    return [[value for _ in range(m)] for _ in range(n)]


def _copy_grid(grid: Grid) -> Grid:
    return [row[:] for row in grid]


def _neighbors(r: int, c: int, n: int, m: int) -> List[Cell]:
    out: List[Cell] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < n:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < m:
        out.append((r, c + 1))
    return out


def _first_empty(grid: Grid) -> Optional[Cell]:
    n, m = len(grid), len(grid[0]) if grid else 0
    for r in range(n):
        for c in range(m):
            if grid[r][c] == 0:
                return (r, c)
    return None


def _count_empty(grid: Grid) -> int:
    return sum(1 for row in grid for x in row if x == 0)


def _partition_possible_dp(max_n: int, sizes: List[int]) -> List[bool]:
    dp = [False] * (max_n + 1)
    dp[0] = True
    for i in range(1, max_n + 1):
        for s in sizes:
            if i - s >= 0 and dp[i - s]:
                dp[i] = True
                break
    return dp


def _random_grow_piece(
    grid: Grid,
    start: Cell,
    size: int,
    rng: random.Random,
    max_internal_steps: int = 200,
) -> Optional[Set[Cell]]:
    """
    Attempt to build a connected set of 'size' empty cells starting at 'start',
    using randomized growth. Returns None if it gets stuck.
    """
    n, m = len(grid), len(grid[0])
    sr, sc = start
    if grid[sr][sc] != 0:
        return None

    piece: Set[Cell] = {(sr, sc)}
    # We allow some retries/steps to avoid getting stuck too easily.
    steps = 0

    while len(piece) < size and steps < max_internal_steps:
        steps += 1
        # Choose a random existing cell and try to expand from it.
        pr, pc = rng.choice(tuple(piece))
        nbrs = _neighbors(pr, pc, n, m)
        rng.shuffle(nbrs)
        added = False
        for nr, nc in nbrs:
            if grid[nr][nc] == 0 and (nr, nc) not in piece:
                piece.add((nr, nc))
                added = True
                break
        if not added:
            # Try another random cell next loop iteration.
            continue

    if len(piece) != size:
        return None
    return piece


def _cells_to_binary_matrix(cells: List[Cell]) -> List[List[int]]:
    """
    Convert a list of (r,c) cells into a tight 0/1 matrix (bounding box).
    """
    if not cells:
        return [[]]
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    rmin, rmax = min(rs), max(rs)
    cmin, cmax = min(cs), max(cs)
    h = rmax - rmin + 1
    w = cmax - cmin + 1
    mat = [[0 for _ in range(w)] for _ in range(h)]
    for r, c in cells:
        mat[r - rmin][c - cmin] = 1
    return mat


def _validate_full_grid(full_grid: Grid, min_size: int, max_size: int) -> None:
    """
    Basic correctness checks:
    - Grid is fully filled (no zeros)
    - Each piece id forms one connected component
    - Each piece size is within [min_size, max_size]
    """
    n = len(full_grid)
    if n == 0 or any(len(row) != n for row in full_grid):
        raise ValueError("full_grid must be a non-empty square grid.")

    if any(x == 0 for row in full_grid for x in row):
        raise ValueError("full_grid contains empty cells (0); expected fully tiled grid.")

    # Map piece_id -> list of cells
    pieces: Dict[int, List[Cell]] = {}
    for r in range(n):
        for c in range(n):
            pid = full_grid[r][c]
            if pid <= 0:
                raise ValueError("full_grid contains non-positive id; expected positive piece ids.")
            pieces.setdefault(pid, []).append((r, c))

    # Check connectivity and sizes
    for pid, cells in pieces.items():
        if not (min_size <= len(cells) <= max_size):
            raise ValueError(f"Piece {pid} has size {len(cells)} outside [{min_size},{max_size}].")

        cellset = set(cells)
        start = cells[0]
        stack = [start]
        seen = {start}
        while stack:
            r, c = stack.pop()
            for nr, nc in _neighbors(r, c, n, n):
                if (nr, nc) in cellset and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
        if len(seen) != len(cells):
            raise ValueError(f"Piece {pid} is not connected.")


def generate_tetris_puzzle(
    size: int = 10,
    min_piece: int = 3,
    max_piece: int = 7,
    clue_fraction_range: Tuple[float, float] = (0.25, 0.55),
    seed: Optional[int] = None,
) -> Tuple[Grid, List[List[List[int]]], Grid]:
    """
    Returns (empty_grid, pieces, full_grid).

    - empty_grid: 0 for removed pieces, numbered ids for clue pieces
    - pieces: list of 0/1 matrices for removed pieces, shuffled
    - full_grid: solution grid with all piece ids

    Robustness:
    - Uses randomized backtracking with multiple restarts.
    - Ensures all pieces are sizes in [min_piece, max_piece] and connected.
    """
    if size <= 0:
        raise ValueError("size must be positive.")
    if not (1 <= min_piece <= max_piece):
        raise ValueError("Invalid piece size bounds.")
    if min_piece == 1 or min_piece == 2:
        # Allowed, but the prompt explicitly wants 3..7 by default.
        pass
    if min_piece > size * size:
        raise ValueError("min_piece is larger than total cells.")
    if not (0.0 <= clue_fraction_range[0] <= clue_fraction_range[1] <= 1.0):
        raise ValueError("clue_fraction_range must be within [0,1] and ordered (low<=high).")

    rng = random.Random(seed)
    sizes = list(range(min_piece, max_piece + 1))
    dp_possible = _partition_possible_dp(size * size, sizes)

    if not dp_possible[size * size]:
        raise ValueError(f"Cannot partition {size*size} cells into sizes {sizes}.")

    # Parameters controlling search effort.
    MAX_RESTARTS = 250
    MAX_SHAPE_ATTEMPTS_PER_SIZE = 60

    def feasible_sizes(remaining: int) -> List[int]:
        # Must be able to complete after placing a piece of size s.
        out = [s for s in sizes if remaining - s >= 0 and dp_possible[remaining - s]]
        rng.shuffle(out)
        return out

    def fill_grid_backtracking(grid: Grid, next_id: int) -> bool:
        remaining = _count_empty(grid)
        if remaining == 0:
            return True

        start = _first_empty(grid)
        if start is None:
            return True

        for s in feasible_sizes(remaining):
            for _ in range(MAX_SHAPE_ATTEMPTS_PER_SIZE):
                cells = _random_grow_piece(grid, start, s, rng)
                if not cells:
                    continue

                # Place piece
                for (r, c) in cells:
                    grid[r][c] = next_id

                if fill_grid_backtracking(grid, next_id + 1):
                    return True

                # Undo piece
                for (r, c) in cells:
                    grid[r][c] = 0

        return False

    full_grid: Optional[Grid] = None

    for _ in range(MAX_RESTARTS):
        g = _make_grid(size, size, 0)
        if fill_grid_backtracking(g, 1):
            # Validate to be safe
            _validate_full_grid(g, min_piece, max_piece)
            full_grid = g
            break

    if full_grid is None:
        raise RuntimeError(
            "Failed to generate a full tiled grid within restart limit. "
            "Try a different seed or increase MAX_RESTARTS."
        )

    # Build piece map from full_grid
    piece_cells: Dict[int, List[Cell]] = {}
    for r in range(size):
        for c in range(size):
            pid = full_grid[r][c]
            piece_cells.setdefault(pid, []).append((r, c))

    all_ids = list(piece_cells.keys())
    rng.shuffle(all_ids)

    # Decide which pieces remain as clues
    # Keep between clue_fraction_range of pieces in-grid as clues.
    low, high = clue_fraction_range
    num_pieces = len(all_ids)
    if num_pieces == 0:
        raise RuntimeError("Unexpected: no pieces generated.")

    target_clues = int(round(num_pieces * rng.uniform(low, high)))
    target_clues = max(1, min(num_pieces - 1, target_clues))  # at least 1 clue and 1 removed piece

    clue_ids = set(all_ids[:target_clues])
    removed_ids = [pid for pid in all_ids if pid not in clue_ids]

    empty_grid = _copy_grid(full_grid)
    for pid in removed_ids:
        for (r, c) in piece_cells[pid]:
            empty_grid[r][c] = 0

    # Convert removed pieces to 0/1 matrices, tight bounding boxes, then shuffle
    pieces: List[List[List[int]]] = []
    for pid in removed_ids:
        pieces.append(_cells_to_binary_matrix(piece_cells[pid]))
    rng.shuffle(pieces)

    return empty_grid, pieces, full_grid


# Example usage:
if __name__ == "__main__":
    empty_grid, pieces, full_grid = generate_tetris_puzzle(seed=None)

    # Variables requested by prompt:
    # - empty_grid
    # - pieces
    # - full_grid

    # Quick sanity prints (optional):
    print("Empty grid (0 = removed):")
    for row in empty_grid:
        print(row)

    print("\nPieces (0/1 matrices):", len(pieces))
    for i, p in enumerate(pieces[:5], 1):
        print(f"\nPiece {i}:")
        for row in p:
            print(row)

    print("\nFull grid (answer):")
    for row in full_grid:
        print(row)
