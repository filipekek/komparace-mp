
def solve_maze(maze):
    """
    Solve a grid maze with:
      0 = open path
      1 = wall
      2 = teleporter (can teleport to any OTHER teleporter)
      3 = quicksand (passable, but entering it adds +1 extra move)

    Start:  (0, 0)
    Finish: (rows-1, cols-1)

    The solver performs a depth-first search with backtracking:
      - Colors the final found path in GREEN
      - Colors explored dead-ends in RED
      - Prints the colored maze and:
          * Correct-path move count (cost along the final path)
          * Total move count (including moves into dead ends and backtracking)

    Assumptions for move costs:
      - Moving to an adjacent cell costs 1 + (destination is quicksand ? 1 : 0).
      - Teleporting from a teleporter to a different teleporter costs 1.
      - Moving back while backtracking also costs based on the destination cell.

    Returns a dict with keys:
      solved (bool), path (list of (r,c)), path_moves (int), total_moves (int)
    """
    # ---------- Validation ----------
    if not isinstance(maze, list) or not maze or not all(isinstance(row, list) for row in maze):
        raise ValueError("maze must be a non-empty list of lists")

    rows = len(maze)
    cols = len(maze[0]) if rows > 0 else 0
    if cols == 0:
        raise ValueError("maze must have at least one column")

    for i, row in enumerate(maze):
        if len(row) != cols:
            raise ValueError("maze must be rectangular (all rows same length)")
        for j, v in enumerate(row):
            if not isinstance(v, int) or v < 0 or v > 3:
                raise ValueError("maze cells must be integers in {0,1,2,3}")

    start = (0, 0)
    finish = (rows - 1, cols - 1)

    def cell(pos):
        r, c = pos
        return maze[r][c]

    def in_bounds(r, c):
        return 0 <= r < rows and 0 <= c < cols

    def passable(pos):
        return cell(pos) != 1

    if not passable(start) or not passable(finish):
        # Print unsolved maze (no path possible if start or finish is a wall)
        _print_colored_maze(maze, path=set(), dead=set())
        print("Correct path moves: 0")
        print("Total moves (including dead ends): 0")
        return {"solved": False, "path": [], "path_moves": 0, "total_moves": 0}

    # Precompute teleporters
    teleporters = []
    for r in range(rows):
        for c in range(cols):
            if maze[r][c] == 2:
                teleporters.append((r, c))

    # ---------- Move cost helpers ----------
    def step_cost(src, dst):
        """Cost of moving from src to dst, either adjacent step or teleport."""
        sr, sc = src
        dr, dc = dst
        # Adjacent move
        if abs(sr - dr) + abs(sc - dc) == 1:
            return 1 + (1 if cell(dst) == 3 else 0)
        # Teleport move (allowed only between teleporters)
        if cell(src) == 2 and cell(dst) == 2 and src != dst:
            return 1
        # Should not happen in a valid traversal
        raise RuntimeError("Internal error: non-adjacent non-teleport move encountered")

    # ---------- DFS with explicit stack (robust vs recursion depth) ----------
    # Track current path (in_path) to avoid cycles; track dead cells to avoid re-exploring proven dead ends.
    in_path = set([start])
    dead = set()
    total_moves = 0

    # Path list parallels the stack of positions.
    path = [start]

    # For efficiency, precompute "other teleporters" list once; we will filter out self at runtime.
    all_teleporters = tuple(teleporters)

    def moves_from(pos):
        """Generate next candidate positions in a deterministic order."""
        r, c = pos

        # Adjacent moves: Right, Down, Left, Up (deterministic)
        candidates = []
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc):
                candidates.append((nr, nc))

        # Teleport moves (if standing on a teleporter)
        if cell(pos) == 2 and len(all_teleporters) > 1:
            # Deterministic ordering: row-major (already)
            for tpos in all_teleporters:
                if tpos != pos:
                    candidates.append(tpos)

        # Filter passability and pruning sets
        for nxt in candidates:
            if nxt in in_path or nxt in dead:
                continue
            if not passable(nxt):
                continue
            yield nxt

    # Stack entries: (pos, iterator_over_moves)
    stack = [(start, moves_from(start))]

    solved = False
    while stack:
        pos, it = stack[-1]

        if pos == finish:
            solved = True
            break

        try:
            nxt = next(it)
        except StopIteration:
            # No more moves from pos -> dead end (unless it's the start; keep it uncolored as dead)
            if pos != start:
                dead.add(pos)

            in_path.remove(pos)
            stack.pop()
            path.pop()

            # Backtrack one step if possible
            if stack:
                back = stack[-1][0]
                total_moves += step_cost(pos, back)
            continue

        # Move forward to nxt
        total_moves += step_cost(pos, nxt)
        in_path.add(nxt)
        path.append(nxt)
        stack.append((nxt, moves_from(nxt)))

    # If solved, in_path currently contains exactly the final path cells.
    path_set = set(path) if solved else set()

    # Compute correct-path move count
    path_moves = 0
    if solved and len(path) >= 2:
        for a, b in zip(path, path[1:]):
            path_moves += step_cost(a, b)

    # Print results
    _print_colored_maze(maze, path=path_set, dead=dead)
    print(f"Correct path moves: {path_moves}")
    print(f"Total moves (including dead ends): {total_moves}")

    # return {"solved": solved, "path": path if solved else [], "path_moves": path_moves, "total_moves": total_moves}


def _print_colored_maze(maze, path, dead):
    """
    Internal helper: print maze with ANSI coloring.
      - path cells in green
      - dead cells in red
    """
    # ANSI colors (foreground)
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"

    # Symbols for each tile type
    symbols = {
        0: ".",
        1: "#",
        2: "T",
        3: "~",
    }

    rows = len(maze)
    cols = len(maze[0]) if rows else 0

    for r in range(rows):
        parts = []
        for c in range(cols):
            ch = symbols.get(maze[r][c], "?")
            pos = (r, c)
            if pos in path:
                parts.append(f"{GREEN}{ch}{RESET}")
            elif pos in dead:
                parts.append(f"{RED}{ch}{RESET}")
            else:
                parts.append(ch)
        print(" ".join(parts))
