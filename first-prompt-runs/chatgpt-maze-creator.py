from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Iterable, List, Optional, Tuple


Tile = int  # 0=open, 1=wall, 2=teleporter, 3=quicksand


@dataclass(frozen=True)
class MazeConfig:
    rows: int = 50
    cols: int = 50
    teleporters: int = 6                 # should be 0 or >= 2 (teleport needs pairs)
    quicksand_ratio: float = 0.02        # fraction of open tiles turned into quicksand
    attempts: int = 20                   # generate multiple mazes; keep the one with longest unique solution
    seed: Optional[int] = None


def generate_maze(cfg: MazeConfig = MazeConfig()) -> List[List[Tile]]:
    """
    Generates a randomized maze on a rows x cols grid of tiles:
      0 = open path
      1 = wall
      2 = teleporter
      3 = quicksand (passable; costs +1 extra move for solvers)

    Properties:
    - Start is at (0, 0), finish is at (rows-1, cols-1)
    - The passable tiles (ignoring teleport jumps) form a tree-like maze (a "perfect maze")
      so there is a unique simple path between any two passable tiles, including start->finish.
    - Teleporters and quicksand are placed on dead ends when possible to avoid creating
      new simple alternative start->finish paths in typical maze-solving interpretations.

    Notes:
    - Internally builds a perfect maze on the odd-coordinate lattice (1..rows-1 step 2),
      carving corridors between those cells. This yields good quality mazes in a tile grid.
    - For even dimensions like 100x100, this still fits cleanly: odd indices go up to 99.
    """
    _validate_config(cfg)

    rng = random.Random(cfg.seed)
    best_grid: List[List[Tile]] | None = None
    best_path_len = -1

    for _ in range(max(1, cfg.attempts)):
        grid = [[1 for _ in range(cfg.cols)] for _ in range(cfg.rows)]
        _carve_perfect_maze_on_odd_lattice(grid, rng)

        # Ensure start/end are open and connected to the carved maze.
        _ensure_start_and_finish_connected(grid)

        # Evaluate how "hard" this maze is by the length of the unique path from start to finish.
        path_len = _unique_path_length_in_tree(grid, (0, 0), (cfg.rows - 1, cfg.cols - 1))
        if path_len > best_path_len:
            best_path_len = path_len
            best_grid = grid

    assert best_grid is not None

    # Place teleporters and quicksand *after* selecting the best structural maze.
    _place_special_tiles(best_grid, rng, teleporters=cfg.teleporters, quicksand_ratio=cfg.quicksand_ratio)

    return best_grid


# ----------------------------- internals ----------------------------- #

def _validate_config(cfg: MazeConfig) -> None:
    if cfg.rows <= 1 or cfg.cols <= 1:
        raise ValueError("rows and cols must be at least 2.")
    if cfg.teleporters < 0:
        raise ValueError("teleporters must be >= 0.")
    if cfg.teleporters == 1:
        raise ValueError("teleporters=1 is invalid; teleporters must be 0 or >= 2.")
    if not (0.0 <= cfg.quicksand_ratio <= 1.0):
        raise ValueError("quicksand_ratio must be within [0.0, 1.0].")
    if cfg.attempts <= 0:
        raise ValueError("attempts must be >= 1.")


def _carve_perfect_maze_on_odd_lattice(grid: List[List[Tile]], rng: random.Random) -> None:
    """
    Builds a perfect maze using iterative randomized DFS on "cells" located at odd coordinates:
      cell positions: (1,1), (1,3), ..., (rows-1 if odd), similarly for columns.

    We carve:
      - each visited cell tile -> open (0)
      - corridor tile between adjacent cells -> open (0)

    This yields a cycle-free passable structure (a tree) when considering 4-neighborhood adjacency.
    """
    rows, cols = len(grid), len(grid[0])

    # Cell lattice dimensions:
    cell_rows = rows // 2  # odd y: 1..(2*cell_rows-1)
    cell_cols = cols // 2  # odd x: 1..(2*cell_cols-1)

    if cell_rows <= 0 or cell_cols <= 0:
        return

    def cell_to_tile(cr: int, cc: int) -> Tuple[int, int]:
        return 2 * cr + 1, 2 * cc + 1

    visited = [[False] * cell_cols for _ in range(cell_rows)]

    start_cr, start_cc = 0, 0
    stack: List[Tuple[int, int]] = [(start_cr, start_cc)]
    visited[start_cr][start_cc] = True
    sr, sc = cell_to_tile(start_cr, start_cc)
    grid[sr][sc] = 0

    # Directions in cell space (move by 1 cell)
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while stack:
        cr, cc = stack[-1]

        # Find unvisited neighbors
        neighbors: List[Tuple[int, int, int, int]] = []
        for dr, dc in dirs:
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < cell_rows and 0 <= nc < cell_cols and not visited[nr][nc]:
                neighbors.append((nr, nc, dr, dc))

        if not neighbors:
            stack.pop()
            continue

        nr, nc, dr, dc = neighbors[rng.randrange(len(neighbors))]
        visited[nr][nc] = True

        # Carve corridor between (cr,cc) and (nr,nc)
        tr1, tc1 = cell_to_tile(cr, cc)
        tr2, tc2 = cell_to_tile(nr, nc)
        mid_r, mid_c = (tr1 + tr2) // 2, (tc1 + tc2) // 2

        grid[mid_r][mid_c] = 0
        grid[tr2][tc2] = 0

        stack.append((nr, nc))


def _ensure_start_and_finish_connected(grid: List[List[Tile]]) -> None:
    """
    Opens (0,0) and (rows-1, cols-1) and connects them to the carved maze if needed.
    With the odd-lattice carving, (1,1) and (rows-1, cols-1) (if odd) are typically open;
    but (0,0) is not part of the lattice and must be connected.
    """
    rows, cols = len(grid), len(grid[0])
    start = (0, 0)
    finish = (rows - 1, cols - 1)

    grid[start[0]][start[1]] = 0
    grid[finish[0]][finish[1]] = 0

    # Connect start to nearest open lattice cell, usually (1,1).
    # We open a small L-shaped connector: (0,1) and (1,0) if in bounds.
    if rows > 1 and cols > 1:
        grid[0][1] = 0
        grid[1][0] = 0
        grid[1][1] = 0  # ensures connection into the carved maze

    # Connect finish if needed: ensure at least one neighbor is open.
    fr, fc = finish
    if _count_open_neighbors(grid, fr, fc) == 0:
        if fr - 1 >= 0:
            grid[fr - 1][fc] = 0
        if fc - 1 >= 0:
            grid[fr][fc - 1] = 0


def _neighbors4(rows: int, cols: int, r: int, c: int) -> Iterable[Tuple[int, int]]:
    if r > 0:
        yield (r - 1, c)
    if r + 1 < rows:
        yield (r + 1, c)
    if c > 0:
        yield (r, c - 1)
    if c + 1 < cols:
        yield (r, c + 1)


def _is_passable(tile: Tile) -> bool:
    return tile in (0, 2, 3)


def _count_open_neighbors(grid: List[List[Tile]], r: int, c: int) -> int:
    rows, cols = len(grid), len(grid[0])
    return sum(1 for nr, nc in _neighbors4(rows, cols, r, c) if _is_passable(grid[nr][nc]))


def _unique_path_length_in_tree(grid: List[List[Tile]], start: Tuple[int, int], goal: Tuple[int, int]) -> int:
    """
    Since the carved maze is (mostly) a tree in 4-neighborhood adjacency,
    the simple path length from start to goal can be computed via BFS parent tracing.
    """
    rows, cols = len(grid), len(grid[0])
    sr, sc = start
    gr, gc = goal

    if not _is_passable(grid[sr][sc]) or not _is_passable(grid[gr][gc]):
        return -1

    parent = [[None] * cols for _ in range(rows)]
    q = deque([start])
    parent[sr][sc] = start  # mark visited

    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            break
        for nr, nc in _neighbors4(rows, cols, r, c):
            if parent[nr][nc] is not None:
                continue
            if not _is_passable(grid[nr][nc]):
                continue
            parent[nr][nc] = (r, c)
            q.append((nr, nc))

    if parent[gr][gc] is None:
        return -1

    # Trace back to count edges
    length = 0
    cur = goal
    while cur != start:
        cur = parent[cur[0]][cur[1]]  # type: ignore[assignment]
        length += 1
    return length


def _place_special_tiles(
    grid: List[List[Tile]],
    rng: random.Random,
    teleporters: int,
    quicksand_ratio: float,
) -> None:
    rows, cols = len(grid), len(grid[0])
    start = (0, 0)
    finish = (rows - 1, cols - 1)

    # Collect passable tiles and dead ends (degree 1 in 4-neighborhood).
    passable: List[Tuple[int, int]] = []
    dead_ends: List[Tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            if not _is_passable(grid[r][c]):
                continue
            if (r, c) in (start, finish):
                continue
            passable.append((r, c))
            if _count_open_neighbors(grid, r, c) == 1:
                dead_ends.append((r, c))

    # Prefer placing specials on dead ends to avoid affecting the main route.
    def pick_positions(k: int, avoid: set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        pool = [p for p in dead_ends if p not in avoid]
        if len(pool) < k:
            pool = [p for p in passable if p not in avoid]
        rng.shuffle(pool)
        return pool[:k]

    used: set[Tuple[int, int]] = set()

    # Teleporters: place an even number (pairing). If odd > 1 is requested, drop the extra.
    if teleporters >= 2:
        if teleporters % 2 == 1:
            teleporters -= 1
        tpos = pick_positions(teleporters, used)
        for r, c in tpos:
            grid[r][c] = 2
            used.add((r, c))

    # Quicksand: convert a fraction of remaining open tiles to quicksand.
    if quicksand_ratio > 0.0:
        # Count current passable tiles (excluding start/finish and excluding teleporters).
        candidates = [
            (r, c)
            for (r, c) in passable
            if (r, c) not in used and grid[r][c] == 0
        ]
        k = int(round(len(candidates) * quicksand_ratio))
        if k > 0:
            # Prefer dead ends first, then others.
            preferred = [p for p in dead_ends if p in candidates]
            others = [p for p in candidates if p not in set(preferred)]
            rng.shuffle(preferred)
            rng.shuffle(others)
            chosen = (preferred + others)[:k]
            for r, c in chosen:
                grid[r][c] = 3

# ----------------------------- example usage ----------------------------- #

if __name__ == "__main__":
    cfg = MazeConfig(seed=000, attempts=30, quicksand_ratio=0.03, teleporters=2)
    maze = generate_maze(cfg)

    # Basic sanity checks
    assert len(maze) == cfg.rows and all(len(row) == cfg.cols for row in maze)
    assert maze[0][0] in (0, 2, 3)
    assert maze[cfg.rows - 1][cfg.cols - 1] in (0, 2, 3)

    # Count tiles (optional)
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for row in maze:
        for v in row:
            counts[v] += 1
    print("Tile counts:", counts)
    print("Unique path length (ignoring teleports/quicksand costs):",
          _unique_path_length_in_tree(maze, (0, 0), (cfg.rows - 1, cfg.cols - 1)))
