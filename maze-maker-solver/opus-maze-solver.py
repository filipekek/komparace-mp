"""
Maze Solver Algorithm

Maze cell types:
    0 = Open path
    1 = Classic wall
    2 = Teleporter (can teleport to any other teleporter)
    3 = Quick sand (passable but adds 1 extra move)

Start: [0][0]
Finish: [-1][-1]

Output: Console visualization with green (correct path) and red (dead ends),
        plus move counts (correct path moves and total moves including dead ends).
"""


def solve_maze(maze):
    """
    Solve the maze and display the solution with colored paths.
    
    Args:
        maze: A list of lists where each element is 0-3 indicating cell type.
              0 = Open path, 1 = Wall, 2 = Teleporter, 3 = Quicksand
    
    Returns:
        A tuple of (solved_maze_display, correct_path_moves, total_moves)
        or None if no solution exists.
    """
    # Validate input
    if not maze or not isinstance(maze, list):
        print("Error: Invalid maze input - maze must be a non-empty list of lists.")
        return None
    
    if not all(isinstance(row, list) for row in maze):
        print("Error: Invalid maze input - each row must be a list.")
        return None
    
    rows = len(maze)
    if rows == 0:
        print("Error: Maze is empty.")
        return None
    
    cols = len(maze[0])
    if cols == 0:
        print("Error: Maze has no columns.")
        return None
    
    # Check all rows have same length
    for i, row in enumerate(maze):
        if len(row) != cols:
            print(f"Error: Row {i} has inconsistent length.")
            return None
    
    # Validate cell values
    for r in range(rows):
        for c in range(cols):
            if maze[r][c] not in (0, 1, 2, 3):
                print(f"Error: Invalid cell value {maze[r][c]} at position ({r}, {c}). Must be 0-3.")
                return None
    
    # Check start and end are accessible (not walls)
    if maze[0][0] == 1:
        print("Error: Start position is a wall.")
        return None
    
    if maze[rows - 1][cols - 1] == 1:
        print("Error: End position is a wall.")
        return None
    
    # Handle trivial case: start equals end
    if rows == 1 and cols == 1:
        _print_maze_colored(maze, {(0, 0)}, set(), rows, cols)
        print(f"\nCorrect path moves: 0")
        print(f"Total moves (including dead ends): 0")
        return
    
    # Find all teleporter positions
    teleporters = []
    for r in range(rows):
        for c in range(cols):
            if maze[r][c] == 2:
                teleporters.append((r, c))
    
    # BFS to find shortest path
    # State: (row, col)
    # Cost tracking considers quicksand (+1 extra)
    
    start = (0, 0)
    end = (rows - 1, cols - 1)
    
    # BFS with cost tracking: (row, col, cost)
    # We use a modified BFS that tracks cost properly for quicksand
    # Since quicksand adds extra cost, we use a priority-like approach with deque
    # For proper shortest path with varying costs, we use a modified Dijkstra-like BFS
    
    # dist[r][c] = minimum cost to reach (r, c)
    dist = [[float('inf')] * cols for _ in range(rows)]
    parent = [[None] * cols for _ in range(rows)]
    visited_order = []  # Track order of exploration for dead end detection
    
    # Starting cost depends on start cell type
    start_cost = 0
    if maze[0][0] == 3:
        start_cost = 1  # Quicksand at start adds 1 extra
    
    dist[0][0] = start_cost
    
    # Use a deque-based approach for 0-1 BFS extended to 0-1-2 costs
    # Since costs are small (0, 1, or 2), we can use buckets
    # But for simplicity and correctness, let's use a proper priority queue simulation
    # with a list of deques for each cost level
    
    # Actually, let's use a simpler approach: modified Dijkstra with deque
    # Since move cost is 1 and quicksand adds 1, costs are 1 or 2
    # We can use 0-1 BFS variant
    
    # Queue entries: (cost, row, col)
    # We'll use a list and always process minimum cost first
    queue = [(start_cost, 0, 0)]
    visited = set()
    exploration_order = []  # All cells explored
    
    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    found = False
    
    while queue:
        # Find minimum cost entry (simple linear search, fine for moderate maze sizes)
        min_idx = 0
        for i in range(1, len(queue)):
            if queue[i][0] < queue[min_idx][0]:
                min_idx = i
        
        cost, r, c = queue.pop(min_idx)
        
        if (r, c) in visited:
            continue
        
        visited.add((r, c))
        exploration_order.append((r, c))
        
        if (r, c) == end:
            found = True
            break
        
        # Get neighbors
        neighbors = []
        
        # Regular movement (4 directions)
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if maze[nr][nc] != 1:  # Not a wall
                    neighbors.append((nr, nc))
        
        # Teleporter movement
        if maze[r][c] == 2:
            for tr, tc in teleporters:
                if (tr, tc) != (r, c):  # Can teleport to other teleporters
                    neighbors.append((tr, tc))
        
        # Process neighbors
        for nr, nc in neighbors:
            if (nr, nc) in visited:
                continue
            
            # Calculate move cost
            move_cost = 1  # Base cost for one move
            if maze[nr][nc] == 3:
                move_cost = 2  # Quicksand adds 1 extra (1 + 1 = 2)
            
            new_cost = cost + move_cost
            
            if new_cost < dist[nr][nc]:
                dist[nr][nc] = new_cost
                parent[nr][nc] = (r, c)
                queue.append((new_cost, nr, nc))
    
    if not found:
        print("No solution found - maze is unsolvable.")
        # Still show the maze with all explored paths as red (dead ends)
        _print_maze_colored(maze, set(), visited - {start}, rows, cols)
        total_moves = _calculate_total_moves(exploration_order, maze)
        print(f"\nCorrect path moves: N/A (no solution)")
        print(f"Total moves (including dead ends): {total_moves}")
        return None
    
    # Reconstruct the correct path
    correct_path = []
    current = end
    while current is not None:
        correct_path.append(current)
        current = parent[current[0]][current[1]]
    correct_path.reverse()
    
    correct_path_set = set(correct_path)
    
    # Dead ends are explored cells not on the correct path (excluding start)
    dead_ends = visited - correct_path_set
    
    # Calculate move counts
    # Correct path moves: count from start to end
    correct_path_moves = dist[end[0]][end[1]]
    
    # Total moves: sum of all exploration moves
    # This counts every cell transition during exploration
    total_moves = _calculate_total_moves(exploration_order, maze)
    
    # Print the colored maze
    _print_maze_colored(maze, correct_path_set, dead_ends, rows, cols)
    
    print(f"\nCorrect path moves: {correct_path_moves}")
    print(f"Total moves (including dead ends): {total_moves}")


def _calculate_total_moves(exploration_order, maze):
    """
    Calculate total moves made during exploration.
    Each cell visited counts as moves based on cell type.
    Start cell quicksand adds 1 extra move.
    Each subsequent cell adds 1 move (or 2 for quicksand).
    """
    if len(exploration_order) == 0:
        return 0
    
    total = 0
    
    # Start cell: if quicksand, adds 1 extra
    start_r, start_c = exploration_order[0]
    if maze[start_r][start_c] == 3:
        total += 1
    
    # Each subsequent cell visited
    for i in range(1, len(exploration_order)):
        r, c = exploration_order[i]
        # Base move cost is 1
        move_cost = 1
        # Quicksand adds 1 extra
        if maze[r][c] == 3:
            move_cost = 2
        total += move_cost
    
    return total


def _print_maze_colored(maze, correct_path, dead_ends, rows, cols):
    """
    Print the maze with ANSI color codes.
    Green for correct path, Red for dead ends.
    """
    # ANSI color codes
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # Cell type symbols
    symbols = {
        0: '.',   # Open path
        1: '#',   # Wall
        2: 'T',   # Teleporter
        3: '~',   # Quicksand
    }
    
    print("\n" + "=" * (cols * 2 + 3))
    print("MAZE SOLUTION")
    print("=" * (cols * 2 + 3))
    print("Legend: . = Path, # = Wall, T = Teleporter, ~ = Quicksand")
    print(f"        {GREEN}Green{RESET} = Correct path, {RED}Red{RESET} = Dead end explored")
    print("-" * (cols * 2 + 3))
    
    for r in range(rows):
        row_str = "| "
        for c in range(cols):
            cell_type = maze[r][c]
            symbol = symbols.get(cell_type, '?')
            
            if (r, c) in correct_path:
                row_str += f"{GREEN}{BOLD}{symbol}{RESET} "
            elif (r, c) in dead_ends:
                row_str += f"{RED}{symbol}{RESET} "
            else:
                row_str += f"{symbol} "
        
        row_str += "|"
        print(row_str)
    
    print("-" * (cols * 2 + 3))