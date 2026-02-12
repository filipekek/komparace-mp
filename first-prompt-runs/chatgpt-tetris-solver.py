from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Dict


Cell = Tuple[int, int]
Grid = List[List[int]]
Piece = List[List[int]]  # 0/1 matrix


@dataclass(frozen=True)
class Placement:
    """One concrete placement of one piece on the board."""
    mask: int                 # bitmask of covered board cells
    cells: Tuple[Cell, ...]   # absolute (r,c) cells covered


def solve_tiling(
    empty_grid: Grid,
    pieces: Sequence[Piece],
    *,
    allow_rotations: bool = True,
    allow_reflections: bool = False,
) -> Optional[Grid]:
    """
    Solve an exact tiling puzzle: place each piece exactly once to fill the grid.

    - empty_grid: list of lists; must be rectangular and contain only 0s.
    - pieces: list of 0/1 matrices; 1s are occupied cells of the piece.
    - allow_rotations: if True, allow 90/180/270-degree rotations of pieces.
    - allow_reflections: if True, also allow mirror reflections.

    Returns a filled grid where piece i is labeled with i+1, or None if no solution exists.
    """

    rows, cols = _validate_empty_grid(empty_grid)
    if rows == 0 or cols == 0:
        # Empty board: solvable only if all pieces have zero area (we disallow that).
        return [] if not pieces else None

    piece_cells: List[Tuple[Cell, ...]] = []
    for p in pieces:
        normalized = _normalize_piece(p)
        cells = _piece_to_cells(normalized)
        if not cells:
            return None  # disallow empty pieces (ambiguous / meaningless)
        piece_cells.append(cells)

    board_area = rows * cols
    total_piece_area = sum(len(c) for c in piece_cells)
    if total_piece_area != board_area:
        return None  # must fill exactly

    # Bit indexing for board: bit = 1 << (r*cols + c)
    full_mask = (1 << board_area) - 1

    # Precompute placements for each piece (considering rotations/reflections if enabled).
    placements_per_piece: List[List[Placement]] = []
    placements_covering_cell: List[Dict[int, List[int]]] = []  # per piece: cell_bit -> [placement_index]

    for idx, original_piece in enumerate(pieces):
        variants = _generate_variants(
            _normalize_piece(original_piece),
            allow_rotations=allow_rotations,
            allow_reflections=allow_reflections,
        )

        # Build unique variants by their cell sets (normalized).
        unique_variant_cells = {_canonical_cells(_piece_to_cells(v)) for v in variants}
        variant_cell_lists = [tuple(sorted(list(cells))) for cells in unique_variant_cells]

        piece_placements: List[Placement] = []
        cover_map: Dict[int, List[int]] = {}

        for cells in variant_cell_lists:
            # cells are relative coords starting near (0,0); compute variant height/width
            max_r = max(r for r, _ in cells)
            max_c = max(c for _, c in cells)
            height, width = max_r + 1, max_c + 1

            for dr in range(rows - height + 1):
                for dc in range(cols - width + 1):
                    abs_cells = tuple((r + dr, c + dc) for r, c in cells)
                    mask = 0
                    for r, c in abs_cells:
                        mask |= 1 << (r * cols + c)
                    # mask always within board by construction

                    placement_index = len(piece_placements)
                    piece_placements.append(Placement(mask=mask, cells=abs_cells))

                    # Map each covered cell bit to this placement index (for fast filtering).
                    m = mask
                    while m:
                        lsb = m & -m
                        cover_map.setdefault(lsb, []).append(placement_index)
                        m ^= lsb

        if not piece_placements:
            return None  # piece cannot fit at all

        placements_per_piece.append(piece_placements)
        placements_covering_cell.append(cover_map)

    # Heuristic: place the most constrained pieces earlier (fewest placements).
    order = sorted(range(len(pieces)), key=lambda i: len(placements_per_piece[i]))
    inv_order = [0] * len(order)
    for pos, original_idx in enumerate(order):
        inv_order[original_idx] = pos

    placements_per_piece = [placements_per_piece[i] for i in order]
    placements_covering_cell = [placements_covering_cell[i] for i in order]

    # Another heuristic: if any piece has zero placements for some board cell requirement later,
    # search will fail naturally; no need for extra checks.

    used = [False] * len(order)
    chosen_placements: List[Optional[int]] = [None] * len(order)  # per ordered piece -> placement index

    def choose_next_empty_cell(occupied: int) -> int:
        # Choose the lowest-index empty cell (fast); can be upgraded to MRV cell selection if desired.
        empty = full_mask ^ occupied
        return empty & -empty  # lowest set bit

    def backtrack(occupied: int, used_count: int) -> bool:
        if occupied == full_mask:
            return used_count == len(order)

        cell_bit = choose_next_empty_cell(occupied)

        # Build candidate moves for this cell: (piece_index, placement_index)
        # We try pieces with fewer ways to cover this specific cell first.
        candidates: List[Tuple[int, int]] = []
        for pi in range(len(order)):
            if used[pi]:
                continue
            plist = placements_covering_cell[pi].get(cell_bit)
            if not plist:
                continue
            # Filter placements that don't overlap current occupancy.
            for pl_idx in plist:
                pl = placements_per_piece[pi][pl_idx]
                if pl.mask & occupied:
                    continue
                candidates.append((pi, pl_idx))

        if not candidates:
            return False

        # Sort candidates to reduce branching: pieces with fewer placements covering this cell first.
        candidates.sort(key=lambda t: len(placements_covering_cell[t[0]].get(cell_bit, [])))

        for pi, pl_idx in candidates:
            pl = placements_per_piece[pi][pl_idx]
            used[pi] = True
            chosen_placements[pi] = pl_idx
            if backtrack(occupied | pl.mask, used_count + 1):
                return True
            chosen_placements[pi] = None
            used[pi] = False

        return False

    if not backtrack(occupied=0, used_count=0):
        return None

    # Build output grid labeled with original piece indices (1..n).
    result = [[0 for _ in range(cols)] for _ in range(rows)]
    for ordered_pi, pl_idx in enumerate(chosen_placements):
        if pl_idx is None:
            return None  # should not happen
        pl = placements_per_piece[ordered_pi][pl_idx]
        original_piece_index = order[ordered_pi]  # map back to original numbering
        label = original_piece_index + 1
        for r, c in pl.cells:
            result[r][c] = label

    return result


# ------------------------ Helpers ------------------------ #

def _validate_empty_grid(grid: Grid) -> Tuple[int, int]:
    if not isinstance(grid, list):
        raise TypeError("empty_grid must be a list of lists")

    if not grid:
        return 0, 0

    if not all(isinstance(row, list) for row in grid):
        raise TypeError("empty_grid must be a list of lists (each row is a list)")

    cols = len(grid[0])
    if any(len(row) != cols for row in grid):
        raise ValueError("empty_grid must be rectangular")

    for row in grid:
        for v in row:
            if v != 0:
                raise ValueError("empty_grid must contain only 0s (empty cells)")

    return len(grid), cols


def _normalize_piece(piece: Piece) -> Piece:
    """Trim all-zero rows/cols around the piece; validate entries are 0/1."""
    if not isinstance(piece, list) or (piece and not all(isinstance(r, list) for r in piece)):
        raise TypeError("each piece must be a list of lists")

    if not piece or not piece[0]:
        return []  # handled as invalid later

    w = len(piece[0])
    if any(len(r) != w for r in piece):
        raise ValueError("each piece must be rectangular")

    for r in piece:
        for v in r:
            if v not in (0, 1):
                raise ValueError("pieces must contain only 0/1 values")

    # Find bounding box of 1s
    ones = [(i, j) for i, row in enumerate(piece) for j, v in enumerate(row) if v == 1]
    if not ones:
        return []

    min_r = min(i for i, _ in ones)
    max_r = max(i for i, _ in ones)
    min_c = min(j for _, j in ones)
    max_c = max(j for _, j in ones)

    return [row[min_c:max_c + 1] for row in piece[min_r:max_r + 1]]


def _piece_to_cells(piece: Piece) -> Tuple[Cell, ...]:
    """Convert a 0/1 matrix to a tuple of (r,c) cells where value==1."""
    if not piece:
        return tuple()
    cells: List[Cell] = []
    for r, row in enumerate(piece):
        for c, v in enumerate(row):
            if v == 1:
                cells.append((r, c))
    return tuple(cells)


def _rotate90(piece: Piece) -> Piece:
    # Rotate clockwise
    return [list(col) for col in zip(*reversed(piece))]


def _reflect(piece: Piece) -> Piece:
    # Mirror horizontally
    return [list(reversed(row)) for row in piece]


def _generate_variants(piece: Piece, *, allow_rotations: bool, allow_reflections: bool) -> List[Piece]:
    """Generate geometric variants of a piece, then normalize each."""
    if not piece:
        return []

    variants: List[Piece] = []
    current = piece
    rotations = 4 if allow_rotations else 1

    for _ in range(rotations):
        variants.append(_normalize_piece(current))
        if allow_reflections:
            variants.append(_normalize_piece(_reflect(current)))
        current = _rotate90(current)

    # Filter out empties created by normalization (should not happen for valid pieces).
    return [v for v in variants if v]


def _canonical_cells(cells: Tuple[Cell, ...]) -> Tuple[Cell, ...]:
    """
    Canonicalize a cell set by translating it so min row/col becomes (0,0),
    then sorting.
    """
    if not cells:
        return tuple()
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    normalized = sorted((r - min_r, c - min_c) for r, c in cells)
    return tuple(normalized)


# ------------------------ Example ------------------------ #

# if __name__ == "__main__":
#     grid = [
#         [0, 0, 0],
#         [0, 0, 0],
#         [0, 0, 0],
#     ]

#     pieces = [
#         [
#             [1, 1, 1],
#             [0, 0, 1],
#         ],
#         [
#             [1, 1],
#             [1, 1],
#         ],
#         [
#             [1],
#         ],
#     ]

#     solved = solve_tiling(grid, pieces, allow_rotations=True, allow_reflections=False)
#     print(solved)
#     # One valid output (matches the prompt’s example) is:
#     # [[1, 1, 1],
#     #  [2, 2, 1],
#     #  [2, 2, 3]]
