import sys

def solve_maze(maze):
    """
    Solves a grid-based maze with specific field types, visualizing the path
    and dead ends, and calculating move costs.
    
    Field Types:
    0: Open path (Cost: 1)
    1: Wall (Impassable)
    2: Teleporter (Cost: 1, connects to all other 2s)
    3: Quick sand (Cost: 2)
    """
    
    # --- 1. Robust Input Validation ---
    if not isinstance(maze, list) or not maze:
        print("Error: Input must be a non-empty list of lists.")
        return
    
    rows = len(maze)
    cols = len(maze[0]) if rows > 0 else 0
    
    if cols == 0:
        print("Error: Maze must have columns.")
        return

    # Ensure the maze is a rectangular grid
    for r in maze:
        if len(r) != cols:
            print("Error: Maze rows must be of equal length.")
            return

    # Check Start and End validity
    start_pos = (0, 0)
    end_pos = (rows - 1, cols - 1)

    if maze[0][0] == 1 or maze[-1][-1] == 1:
        print("Error: Start or Finish is a wall.")
        return

    # Increase recursion limit for deep mazes (DFS depth)
    sys.setrecursionlimit(max(1000, rows * cols + 100))

    # --- 2. Pre-processing ---
    # Find all teleporters to allow O(1) access
    teleporters = []
    for r in range(rows):
        for c in range(cols):
            if maze[r][c] == 2:
                teleporters.append((r, c))

    # --- 3. DFS Solver State ---
    # Path: List of coordinates in the current successful route
    path = []
    # Visited: Set of all coordinates entered
    visited = set()
    # Dead Ends: Visited nodes that are NOT in the final path
    dead_ends = set()
    
    # Counters
    # We use a mutable container (list) to allow modification inside inner function
    metrics = {
        "total_attempted_cost": 0,
        "final_path_cost": 0
    }

    def get_move_cost(val):
        """Returns cost to enter a cell with specific value."""
        if val == 3: return 2  # Quicksand
        return 1               # Open or Teleporter

    def get_neighbors(r, c):
        """Generates valid moves from current position."""
        moves = []
        
        # Standard neighbors (Up, Down, Left, Right)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if maze[nr][nc] != 1: # Not a wall
                    moves.append((nr, nc))
        
        # Teleportation logic
        # If current spot is a teleporter, we can go to any OTHER teleporter
        if maze[r][c] == 2:
            for tr, tc in teleporters:
                if (tr, tc) != (r, c):
                    moves.append((tr, tc))
                    
        return moves

    # --- 4. The Algorithm (DFS) ---
    def dfs(r, c):
        # 1. Mark current node
        visited.add((r, c))
        path.append((r, c))
        
        # 2. Check for Win Condition
        if (r, c) == end_pos:
            return True
        
        # 3. Explore Neighbors
        for nr, nc in get_neighbors(r, c):
            if (nr, nc) not in visited:
                # Calculate cost to enter this neighbor
                move_cost = get_move_cost(maze[nr][nc])
                
                # Update Total Moves (Exploration Cost)
                metrics["total_attempted_cost"] += move_cost
                
                if dfs(nr, nc):
                    # If this branch leads to success, add to Final Path Cost
                    metrics["final_path_cost"] += move_cost
                    return True
        
        # 4. Backtrack (Dead End)
        # If no neighbors led to the end, remove from path
        path.pop()
        return False

    # --- 5. Execution ---
    success = dfs(start_pos[0], start_pos[1])

    # Determine dead ends: All visited nodes that aren't in the final path
    path_set = set(path)
    dead_ends = visited - path_set

    # --- 6. Visualization & Output ---
    # ANSI Color Codes
    GREEN = '\033[92m'  # Correct Path
    RED = '\033[91m'    # Dead Ends
    BLUE = '\033[94m'   # Teleporter/Special
    RESET = '\033[0m'
    
    print("\n" + "="*30)
    print(f"MAZE SOLVER RESULTS")
    print("="*30 + "\n")

    if not success:
        print(f"{RED}No path found from Start to Finish.{RESET}")
        print(f"Total exploration cost: {metrics['total_attempted_cost']}")
        return

    # Render Grid
    output_lines = []
    
    # Legend mapping for visualization
    # 0 -> ' ' or '.'
    # 1 -> '█' (Wall)
    # 2 -> 'T'
    # 3 -> '~' (Sand)
    
    for r in range(rows):
        line = ""
        for c in range(cols):
            cell_val = maze[r][c]
            char = "?"
            
            # Base character based on type
            if cell_val == 0: char = " "
            elif cell_val == 1: char = "█"
            elif cell_val == 2: char = "T"
            elif cell_val == 3: char = "~"
            
            # Apply Color Overlays
            if (r, c) == start_pos:
                line += f"{GREEN}S{RESET} "
            elif (r, c) == end_pos:
                line += f"{GREEN}E{RESET} "
            elif (r, c) in path_set:
                # Correct Path
                line += f"{GREEN}•{RESET} "
            elif (r, c) in dead_ends:
                # Dead End
                line += f"{RED}x{RESET} "
            else:
                # Unvisited or Walls
                line += f"{char} "
        output_lines.append(line)

    # Print the Map
    print(f"Legend: {GREEN}S/E/•{RESET}=Path, {RED}x{RESET}=Dead End, █=Wall, T=Teleporter, ~=Sand\n")
    for line in output_lines:
        print(line)

    # Print Statistics
    print("\n" + "-"*30)
    print(f"Status:            {GREEN}SOLVED{RESET}")
    print(f"Correct Moves:     {metrics['final_path_cost']}")
    print(f"Total Moves:       {metrics['total_attempted_cost']}")
    print("-"*30 + "\n")