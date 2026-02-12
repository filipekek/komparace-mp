from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

Coord = Tuple[int, int]  # (row, col)
Grid = List[List[int]]


# -----------------------------
# Tetromino definitions
# -----------------------------
# Base shapes as sets of (r, c) coordinates, anchored at (0,0) somewhere in the piece.
# We'll generate all unique orientations (rotations + reflections) from these bases.
BASE_TETROMINOES: Dict[str, Set[Coord]] = {
    "I": {(0, 0), (0, 1), (0, 2), (0, 3)},
    "O": {(0, 0), (0, 1), (1, 0), (1, 1)},
    "T": {(0, 0), (0, 1), (0, 2), (1, 1)},
    "S": {(0, 1), (0, 2), (1, 0), (1, 1)},
    "Z": {(0, 0), (0, 1), (1, 1), (1, 2)},
    "J": {(0, 0), (1, 0), (1, 1), (1, 2)},
    "L": {(0, 2), (1, 0), (1, 1), (1, 2)},
}


def _normalize(shape: Iterable[Coord]) -> Tuple[Coord, ...]:
    """Shift so min row/col become (0,0); return sorted tuple for hashing/comparison."""
    pts = list(shape)
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    norm = sorted((r - min_r, c - min_c) for r, c in pts)
    return tuple(norm)


def _rot90(shape: Iterable[Coord]) -> List[Coord]:
    """Rotate 90 degrees clockwise around origin."""
    pts = list(shape)
    # (r, c) -> (c, -r)
    return [(c, -r) for r, c in pts]


def _reflect_h(shape: Iterable[Coord]) -> List[Coord]:
    """Reflect horizontally (mirror across vertical axis)."""
    pts = list(shape)
    # (r, c) -> (r, -c)
    return [(r, -c) for r, c in pts]


def generate_unique_orientations(base: Set[Coord]) -> List[Tuple[Coord, ...]]:
    """All unique orientations (rotations + reflections), normalized."""
    seen: Set[Tuple[Coord, ...]] = set()
    out: List[Tuple[Coord, ...]] = []

    variants: List[List[Coord]] = [list(base)]
    # 3 more rotations
    for _ in range(3):
        variants.append(_rot90(variants[-1]))
    # add reflected versions + their rotations
    reflected = _reflect_h(base)
    variants.append(reflected)
    for _ in range(3):
        variants.append(_rot90(variants[-1]))

    for v in variants:
        key = _normalize(v)
        if key not in seen:
            seen.add(key)
            out.append(key)

    return out


ORIENTATIONS: Dict[str, List[Tuple[Coord, ...]]] = {
    name: generate_unique_orientations(shape) for name, shape in BASE_TETROMINOES.items()
}


# -----------------------------
# Puzzle pieces representation
# -----------------------------
@dataclass(frozen=True)
class Piece:
    """A placed piece, identified by an integer id, with absolute grid cells."""
    pid: int
    kind: str
    cells: Tuple[Coord, ...]  # absolute (r,c)


def piece_to_minimal_grid(piece: Piece) -> List[List[int]]:
    """
    Convert piece cells into a minimal bounding-box grid filled with 1s.
    This is a compact, readable representation for "pulled out" pieces.
    """
    rs = [r for r, _ in piece.cells]
    cs = [c for _, c in piece.cells]
    min_r, max_r = min(rs), max(rs)
    min_c, max_c = min(cs), max(cs)
    h = max_r - min_r + 1
    w = max_c - min_c + 1
    g = [[0] * w for _ in range(h)]
    for r, c in piece.cells:
        g[r - min_r][c - min_c] = 1
    return g


# -----------------------------
# Grid filling (randomized backtracking)
# -----------------------------
def make_empty_grid(n: int = 10) -> Grid:
    if n <= 0:
        raise ValueError("Grid size must be positive.")
    return [[0 for _ in range(n)] for _ in range(n)]


def _find_first_empty(grid: Grid) -> Optional[Coord]:
    n = len(grid)
    for r in range(n):
        row = grid[r]
        for c in range(n):
            if row[c] == 0:
                return (r, c)
    return None


def _can_place(grid: Grid, top: int, left: int, shape: Sequence[Coord]) -> bool:
    n = len(grid)
    for dr, dc in shape:
        r, c = top + dr, left + dc
        if r < 0 or r >= n or c < 0 or c >= n or grid[r][c] != 0:
            return False
    return True


def _place(grid: Grid, pid: int, top: int, left: int, shape: Sequence[Coord]) -> Tuple[Coord, ...]:
    cells: List[Coord] = []
    for dr, dc in shape:
        r, c = top + dr, left + dc
        grid[r][c] = pid
        cells.append((r, c))
    return tuple(cells)


def _unplace(grid: Grid, cells: Sequence[Coord]) -> None:
    for r, c in cells:
        grid[r][c] = 0


def _candidate_placements_for_cell(r: int, c: int, n: int) -> List[Tuple[int, int]]:
    """
    For a given cell, return candidate (top,left) anchors to try.
    Keeping this small improves performance.
    """
    # We will anchor shapes at or near (r,c) by letting (r,c) correspond to any shape cell.
    # This is handled in the solver by computing (top,left) from a chosen shape cell.
    return [(r, c)]  # placeholder; actual anchors computed per-shape-cell


def generate_tetris_puzzle(
    n: int = 10,
    *,
    seed: Optional[int] = None,
    max_restarts: int = 50,
) -> Tuple[Grid, List[List[List[int]]], Grid]:
    """
    Create a random n x n tetris-style tiling using tetrominoes.

    Returns:
      empty_grid: an n x n grid of zeros
      pieces: a list of pieces in random order, each as a minimal 0/1 grid (1 = occupied)
      full_grid: the completed n x n grid with unique integer ids per piece

    Notes on robustness:
      - Requires n*n divisible by 4 (tetromino area).
      - Uses randomized backtracking with restarts to ensure it finds a solution.
      - For n=10, this is typically fast.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if (n * n) % 4 != 0:
        raise ValueError(f"n*n must be divisible by 4 for tetromino tiling. Got n={n} -> {n*n} cells.")

    rng = random.Random(seed)

    # Pre-build a randomized "bag" of kinds to bias variety.
    kinds = list(ORIENTATIONS.keys())

    def attempt_once() -> Optional[Tuple[Grid, List[Piece]]]:
        grid = make_empty_grid(n)
        next_pid = 1
        placed: List[Piece] = []

        # A small heuristic: pre-shuffle orientation lists per kind each attempt.
        orientations_shuffled: Dict[str, List[Tuple[Coord, ...]]] = {
            k: rng.sample(ORIENTATIONS[k], k=len(ORIENTATIONS[k])) for k in kinds
        }

        # Another heuristic: choose kind order randomly at each decision.
        def backtrack() -> bool:
            nonlocal next_pid

            empty = _find_first_empty(grid)
            if empty is None:
                return True  # filled

            r0, c0 = empty

            # Randomize kinds each step for variety.
            step_kinds = rng.sample(kinds, k=len(kinds))

            for kind in step_kinds:
                for shape in orientations_shuffled[kind]:
                    # Try aligning each cell of the shape onto the chosen empty cell.
                    # That yields different possible (top,left) anchors.
                    for sr, sc in shape:
                        top = r0 - sr
                        left = c0 - sc
                        if _can_place(grid, top, left, shape):
                            cells = _place(grid, next_pid, top, left, shape)
                            placed.append(Piece(pid=next_pid, kind=kind, cells=cells))
                            next_pid += 1

                            if backtrack():
                                return True

                            # undo
                            last = placed.pop()
                            _unplace(grid, last.cells)
                            next_pid -= 1

            return False

        if backtrack():
            return grid, placed
        return None

    # Random restarts (cheap insurance for unlucky random choices)
    for _ in range(max_restarts):
        res = attempt_once()
        if res is not None:
            full_grid, placed_pieces = res

            # Build the pulled-out pieces in a good representation:
            # minimal bounding-box 0/1 grids (compact and readable),
            # then shuffle their order.
            pieces_grids = [piece_to_minimal_grid(p) for p in placed_pieces]
            rng.shuffle(pieces_grids)

            return make_empty_grid(n), pieces_grids, full_grid

    raise RuntimeError(
        f"Failed to generate a tiling after {max_restarts} restarts. "
        "Try increasing max_restarts or changing the seed."
    )


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    empty, pieces, answer = generate_tetris_puzzle(n=10, seed=None, max_restarts=50)

    print("Empty grid:")
    for row in empty:
        print(row)

    print("\nPieces (shuffled), each as a minimal 0/1 grid:")
    for i, p in enumerate(pieces, 1):
        print(f"\nPiece #{i}:")
        for row in p:
            print(row)

    print("\nFull grid (answer):")
    for row in answer:
        print(row)
