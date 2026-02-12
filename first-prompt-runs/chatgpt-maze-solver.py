from __future__ import annotations
from typing import List, Tuple, Set, Dict, Optional

# Cell types
OPEN = 0
WALL = 1
TELEPORT = 2
QUICKSAND = 3

# Change this if you want teleporting itself to count as a move.
TELEPORT_COST = 0

# ANSI colors (works in most terminals)
GREEN = "\033[92m"
RED = "\033[91m"
DIM = "\033[2m"
RESET = "\033[0m"


Pos = Tuple[int, int]


def solve_maze(maze: List[List[int]]) -> None:
    """
    Solves the maze by actually walking it (DFS + backtracking).
    Prints the final colored maze and the total move count.

    Maze rules:
      - Start: (0, 0)
      - Finish: (rows-1, cols-1)
      - 0 open: passable, step cost 1
      - 1 wall: impassable
      - 2 teleporter: passable, step cost 1, can teleport to any other teleporter (cost TELEPORT_COST)
      - 3 quicksand: passable, step cost 1 + extra 1 (total 2) when stepping into it
    """
    _validate_maze(maze)

    rows, cols = len(maze), len(maze[0])
    start = (0, 0)
    goal = (rows - 1, cols - 1)

    if maze[start[0]][start[1]] == WALL:
        raise ValueError("Start cell [0][0] is a wall.")
    if maze[goal[0]][goal[1]] == WALL:
        raise ValueError("Finish cell [-1][-1] is a wall.")

    teleporters = _collect_teleporters(maze)

    visited: Set[Pos] = set()
    path_stack: List[Pos] = []
    dead_ends: Set[Pos] = set()
    entered: Set[Pos] = set()  # all cells the solver actually stepped into (including backtracks)

    total_moves = 0

    # To count moves realistically, we count both forward moves and backtrack moves.
    # We also apply quicksand extra cost when stepping INTO quicksand (even during backtracking).
    def step_cost(dst: Pos) -> int:
        r, c = dst
        return 1 + (1 if maze[r][c] == QUICKSAND else 0)

    # Neighbor order chosen to be deterministic and easy to understand.
    # (Right, Down, Left, Up)
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def in_bounds(p: Pos) -> bool:
        r, c = p
        return 0 <= r < rows and 0 <= c < cols

    def passable(p: Pos) -> bool:
        r, c = p
        return maze[r][c] != WALL

    def dfs(cur: Pos) -> bool:
        nonlocal total_moves

        visited.add(cur)
        path_stack.append(cur)

        if cur == goal:
            return True

        r, c = cur

        # Build candidate next positions: adjacent + (optional) teleports if on teleporter.
        candidates: List[Tuple[Pos, int]] = []

        # Adjacent moves
        for dr, dc in directions:
            nxt = (r + dr, c + dc)
            if in_bounds(nxt) and passable(nxt) and nxt not in visited:
                candidates.append((nxt, step_cost(nxt)))

        # Teleport moves (only if standing on a teleporter)
        if maze[r][c] == TELEPORT and len(teleporters) >= 2:
            for t in teleporters:
                if t != cur and t not in visited:
                    candidates.append((t, TELEPORT_COST))

        # Try each candidate
        for nxt, move_cost in candidates:
            # Perform the move
            total_moves += move_cost
            entered.add(nxt)

            if dfs(nxt):
                return True

            # Backtrack: move back to cur
            # If we teleported (move_cost could be 0/1), backtracking is also a move:
            # - If we teleported: backtrack teleport cost = TELEPORT_COST
            # - If we walked: backtrack cost is based on stepping back INTO cur (quicksand rule applies)
            if _is_teleport_move(cur, nxt, maze):
                total_moves += TELEPORT_COST
            else:
                total_moves += step_cost(cur)

        # If no candidate worked, cur is a dead end (unless it is start)
        path_stack.pop()
        if cur != start:
            dead_ends.add(cur)
        return False

    entered.add(start)
    found = dfs(start)

    if not found:
        print("No path found.")
        _print_maze(maze, set(), dead_ends, start, goal)
        print(f"Moves: {total_moves}")
        return

    final_path = set(path_stack)  # remaining stack is the successful route
    _print_maze(maze, final_path, dead_ends, start, goal)
    print(f"Moves: {total_moves}")


def _is_teleport_move(a: Pos, b: Pos, maze: List[List[int]]) -> bool:
    # Teleport move if both are teleporters and they are not adjacent (or even if adjacent, treat as teleport only if chosen as such).
    # In our solver, we add teleports only from a teleporter cell.
    ar, ac = a
    br, bc = b
    if maze[ar][ac] != TELEPORT or maze[br][bc] != TELEPORT:
        return False
    manhattan = abs(ar - br) + abs(ac - bc)
    # If far apart, it's definitely teleport. If adjacent, it could also be a normal move;
    # we treat adjacent teleporter-to-teleporter as normal walking, because it appears in adjacency list.
    return manhattan > 1


def _collect_teleporters(maze: List[List[int]]) -> List[Pos]:
    tele = []
    for r in range(len(maze)):
        for c in range(len(maze[0])):
            if maze[r][c] == TELEPORT:
                tele.append((r, c))
    return tele


def _validate_maze(maze: List[List[int]]) -> None:
    if not isinstance(maze, list) or not maze or not all(isinstance(row, list) for row in maze):
        raise ValueError("Maze must be a non-empty list of lists.")
    width = len(maze[0])
    if width == 0:
        raise ValueError("Maze rows must be non-empty.")
    for row in maze:
        if len(row) != width:
            raise ValueError("All maze rows must have the same length.")
        for cell in row:
            if cell not in (OPEN, WALL, TELEPORT, QUICKSAND):
                raise ValueError("Maze values must be 0, 1, 2, or 3.")


def _print_maze(
    maze: List[List[int]],
    path: Set[Pos],
    dead_ends: Set[Pos],
    start: Pos,
    goal: Pos,
) -> None:
    rows, cols = len(maze), len(maze[0])

    def symbol(r: int, c: int) -> str:
        if (r, c) == start:
            return "S"
        if (r, c) == goal:
            return "F"
        v = maze[r][c]
        if v == WALL:
            return "#"
        if v == OPEN:
            return "."
        if v == TELEPORT:
            return "T"
        if v == QUICKSAND:
            return "Q"
        return "?"

    for r in range(rows):
        line_parts = []
        for c in range(cols):
            p = (r, c)
            ch = symbol(r, c)

            if maze[r][c] == WALL:
                line_parts.append(DIM + ch + RESET)
            elif p in path and p not in (start, goal):
                line_parts.append(GREEN + ch + RESET)
            elif p in dead_ends and p not in (start, goal):
                line_parts.append(RED + ch + RESET)
            else:
                line_parts.append(ch)
        print(" ".join(line_parts))


# -------------------------
# Example usage (you replace the maze with your own)
# -------------------------
# if __name__ == "__main__":
#     # User-provided maze example:
#     maze = [
#         [0, 0, 1, 0, 2],
#         [1, 0, 1, 0, 0],
#         [0, 0, 0, 3, 1],
#         [2, 1, 0, 0, 0],
#         [0, 0, 1, 1, 0],
#     ]
#     solve_maze(maze)
