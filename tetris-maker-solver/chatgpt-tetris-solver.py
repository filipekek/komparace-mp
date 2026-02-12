from typing import List, Tuple, Dict, Iterable, Optional


def solve_puzzle(grid: List[List[int]], pieces: List[List[List[int]]]) -> int:
    """
    Count the number of ways to fill all 0-cells of a square grid with the given pieces,
    without covering any pre-filled (non-zero) cells.

    - grid: square list-of-lists of ints; 0 means empty, non-zero means fixed/blocked.
    - pieces: list of 0/1 matrices (list-of-lists), each representing a piece.
      Pieces may be rotated (0/90/180/270) and may be mirrored (reflection), producing
      unique orientations.

    Returns:
        Number of exact tilings (solutions) as an int.
    """

    # ----------------------------
    # Input validation
    # ----------------------------
    if not isinstance(grid, list) or not grid:
        return 0
    n = len(grid)
    if any(not isinstance(row, list) or len(row) != n for row in grid):
        return 0  # must be square
    for row in grid:
        for v in row:
            if not isinstance(v, int):
                return 0

    if pieces is None:
        return 0
    if not isinstance(pieces, list):
        return 0

    # ----------------------------
    # Build fixed + free masks
    # ----------------------------
    def cell_index(r: int, c: int) -> int:
        return r * n + c

    fixed_mask = 0
    free_mask = 0
    used_numbers = set()
    for r in range(n):
        for c in range(n):
            v = grid[r][c]
            if v != 0:
                fixed_mask |= 1 << cell_index(r, c)
                used_numbers.add(v)
            else:
                free_mask |= 1 << cell_index(r, c)

    free_cells = _popcount(free_mask)

    # Quick edge case: nothing to fill
    if free_cells == 0:
        return 1 if len(pieces) == 0 else 0

    # ----------------------------
    # Piece normalization + orientation generation
    # ----------------------------
    def validate_piece_matrix(mat: List[List[int]]) -> bool:
        if not isinstance(mat, list) or not mat:
            return False
        if any(not isinstance(row, list) or not row for row in mat):
            return False
        w = len(mat[0])
        if any(len(row) != w for row in mat):
            return False
        for row in mat:
            for x in row:
                if x not in (0, 1):
                    return False
        return True

    def coords_from_matrix(mat: List[List[int]]) -> List[Tuple[int, int]]:
        coords = []
        for rr, row in enumerate(mat):
            for cc, x in enumerate(row):
                if x == 1:
                    coords.append((rr, cc))
        return coords

    def normalize_coords(coords: Iterable[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
        coords = list(coords)
        min_r = min(r for r, _ in coords)
        min_c = min(c for _, c in coords)
        norm = sorted((r - min_r, c - min_c) for r, c in coords)
        return tuple(norm)

    def rotate90(coords: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
        # Rotate around origin: (r, c) -> (c, -r), then normalize later
        coords = list(coords)
        return [(c, -r) for r, c in coords]

    def reflect(coords: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
        # Mirror across vertical axis: (r, c) -> (r, -c), then normalize later
        coords = list(coords)
        return [(r, -c) for r, c in coords]

    def unique_orientations(mat: List[List[int]]) -> List[Tuple[Tuple[int, int], ...]]:
        base = coords_from_matrix(mat)
        if not base:
            return []
        seen = set()
        result = []

        for do_reflect in (False, True):
            current = base
            if do_reflect:
                current = reflect(current)

            # Apply 4 rotations
            rot = current
            for _ in range(4):
                norm = normalize_coords(rot)
                if norm not in seen:
                    seen.add(norm)
                    result.append(norm)
                rot = rotate90(rot)

        return result

    # Validate pieces and compute total area
    piece_orients: List[List[Tuple[Tuple[int, int], ...]]] = []
    piece_areas: List[int] = []

    for p in pieces:
        if not validate_piece_matrix(p):
            return 0
        orients = unique_orientations(p)
        if not orients:
            return 0  # empty piece is not usable for exact fill
        piece_orients.append(orients)
        # Area is count of 1s in any orientation (same for all)
        piece_areas.append(len(orients[0]))

    if sum(piece_areas) != free_cells:
        return 0

    # ----------------------------
    # Precompute all valid placements per piece (respecting fixed cells)
    # ----------------------------
    placements_per_piece: List[List[int]] = []

    for orients in piece_orients:
        all_masks = []
        seen_masks = set()
        for orient in orients:
            max_r = max(r for r, _ in orient)
            max_c = max(c for _, c in orient)
            h = max_r + 1
            w = max_c + 1

            if h > n or w > n:
                continue

            for r0 in range(n - h + 1):
                for c0 in range(n - w + 1):
                    m = 0
                    ok = True
                    for dr, dc in orient:
                        rr = r0 + dr
                        cc = c0 + dc
                        bit = 1 << cell_index(rr, cc)
                        if fixed_mask & bit:
                            ok = False
                            break
                        m |= bit
                    if ok:
                        # Must cover only free cells eventually; but we allow placement over
                        # cells that are currently free_mask by construction (fixed checked).
                        # Later, recursion ensures exact coverage with remaining cells.
                        if m not in seen_masks:
                            seen_masks.add(m)
                            all_masks.append(m)

        if not all_masks:
            return 0
        placements_per_piece.append(all_masks)

    num_pieces = len(placements_per_piece)
    all_unused = (1 << num_pieces) - 1

    # ----------------------------
    # Build cell -> options index (accelerates branching)
    # Each option is (piece_index, placement_mask)
    # Only for cells that are in free_mask.
    # ----------------------------
    cell_to_options: List[List[Tuple[int, int]]] = [[] for _ in range(n * n)]
    for pi, plist in enumerate(placements_per_piece):
        for pmask in plist:
            # For every covered cell, register this placement as an option for that cell.
            mm = pmask
            while mm:
                lsb = mm & -mm
                idx = (lsb.bit_length() - 1)
                cell_to_options[idx].append((pi, pmask))
                mm ^= lsb

    # Some free cells might be unreachable (no placement covers them)
    mm = free_mask
    while mm:
        lsb = mm & -mm
        idx = lsb.bit_length() - 1
        if not cell_to_options[idx]:
            return 0
        mm ^= lsb

    # ----------------------------
    # Exact cover search with memoization
    # State: (remaining_cells_mask, unused_pieces_mask)
    # ----------------------------
    memo: Dict[Tuple[int, int], int] = {}

    def select_next_cell(rem_mask: int, unused_mask: int) -> Optional[int]:
        """
        Choose a remaining cell with the smallest number of viable options under current unused pieces.
        Returns cell index, or None if rem_mask == 0.
        """
        if rem_mask == 0:
            return None

        best_idx = None
        best_count = None

        m = rem_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1

            # Count viable options for this cell
            cnt = 0
            for pi, pmask in cell_to_options[idx]:
                if (unused_mask >> pi) & 1:
                    if (pmask & rem_mask) == pmask:
                        cnt += 1
                        # Small early cut: if already worse than current best, stop counting
                        if best_count is not None and cnt >= best_count:
                            break

            if cnt == 0:
                return idx  # dead end fast
            if best_count is None or cnt < best_count:
                best_count = cnt
                best_idx = idx
                if best_count == 1:
                    # Can't do better than 1
                    pass

            m ^= lsb

        return best_idx

    def dfs(rem_mask: int, unused_mask: int) -> int:
        if rem_mask == 0:
            return 1 if unused_mask == 0 else 0

        key = (rem_mask, unused_mask)
        if key in memo:
            return memo[key]

        idx = select_next_cell(rem_mask, unused_mask)
        if idx is None:
            memo[key] = 0
            return 0

        total = 0
        for pi, pmask in cell_to_options[idx]:
            if ((unused_mask >> pi) & 1) == 0:
                continue
            if (pmask & rem_mask) != pmask:
                continue

            total += dfs(rem_mask ^ pmask, unused_mask & ~(1 << pi))

        memo[key] = total
        return total

    return dfs(free_mask, all_unused)


def _popcount(x: int) -> int:
    # Python 3.8+ has int.bit_count(); keep robust fallback.
    try:
        return x.bit_count()
    except AttributeError:
        c = 0
        while x:
            x &= x - 1
            c += 1
        return c
