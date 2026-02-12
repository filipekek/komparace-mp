import random
import time

def generate_maze():
    """
    Generates a 50x50 randomized maze with specific constraints.
    0 = Open path, 1 = Wall, 2 = Teleporter, 3 = Quicksand.
    Returns: list of lists (the maze).
    """
    width = 50
    height = 50
    
    # Initialize grid with Walls (1)
    maze = [[1 for _ in range(width)] for _ in range(height)]
    
    # ---------------------------------------------------------
    # 1. MAZE GENERATION (Iterative Backtracker)
    # ---------------------------------------------------------
    # We carve paths (0) into the walls. 
    # We use a stack to act as recursive calls (prevents recursion depth errors).
    # We step 2 cells at a time to maintain walls between paths.
    
    start_x, start_y = 0, 0
    maze[start_y][start_x] = 0
    
    stack = [(start_x, start_y)]
    
    while stack:
        x, y = stack[-1]
        
        # Define possible directions: (dx, dy)
        # We look 2 steps away
        directions = [(0, -2), (0, 2), (-2, 0), (2, 0)]
        random.shuffle(directions)
        
        carved = False
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            # Check if neighbor is within bounds and is currently a Wall
            if 0 <= nx < width and 0 <= ny < height and maze[ny][nx] == 1:
                # Carve the target cell
                maze[ny][nx] = 0
                # Carve the wall strictly between current and target
                maze[y + dy // 2][x + dx // 2] = 0
                
                stack.append((nx, ny))
                carved = True
                break
        
        if not carved:
            stack.pop()

    # ---------------------------------------------------------
    # 2. HANDLE EVEN GRID DIMENSIONS & EXIT
    # ---------------------------------------------------------
    # Because the grid is even (50x50) and we step by 2 starting at 0,
    # the backtracker covers indices 0, 2, ... 48. 
    # Row 49 and Col 49 might remain walls. We must ensure the exit (49, 49) is open
    # and connected to the maze structure.
    
    # Force open the destination
    dest_x, dest_y = width - 1, height - 1
    maze[dest_y][dest_x] = 0
    
    # Connect destination to the nearest open path if it's blocked
    # Check neighbors of destination
    connected = False
    if maze[dest_y][dest_x-1] == 0 or maze[dest_y-1][dest_x] == 0:
        connected = True
    
    # If not connected naturally, force a connection to (48, 49) or (49, 48)
    if not connected:
        # Prefer connecting to the left or up
        if maze[dest_y][dest_x - 1] == 1:
            maze[dest_y][dest_x - 1] = 0 # Carve Left
        elif maze[dest_y - 1][dest_x] == 1:
            maze[dest_y - 1][dest_x] = 0 # Carve Up

    # ---------------------------------------------------------
    # 3. IDENTIFY SOLUTION PATH
    # ---------------------------------------------------------
    # To maintain "One Possible Correct Path", we must identify the path
    # and ensure Teleporters do not create shortcuts on it.
    # We run a BFS to find the path coordinates.
    
    queue = [[(0, 0)]] # Queue stores paths
    visited = set([(0, 0)])
    solution_path_set = set()
    
    found_path = False
    while queue:
        path = queue.pop(0)
        cx, cy = path[-1]
        
        if cx == dest_x and cy == dest_y:
            solution_path_set = set(path)
            found_path = True
            break
        
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < width and 0 <= ny < height:
                if maze[ny][nx] == 0 and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    new_path = list(path)
                    new_path.append((nx, ny))
                    queue.append(new_path)
    
    # Fallback (rare): If heuristic connection failed, just carve a line (should not happen with logic above)
    if not found_path:
        # Force carve straight line (emergency fallback to ensure validity)
        curr_x, curr_y = 0, 0
        while curr_x < 49:
            curr_x += 1
            maze[0][curr_x] = 0
        while curr_y < 49:
            curr_y += 1
            maze[curr_y][49] = 0
        solution_path_set = set() # Logic below will skip path protection if this happens

    # ---------------------------------------------------------
    # 4. ADD FEATURES (Teleporters & Quicksand)
    # ---------------------------------------------------------
    
    # Find all open cells that are NOT start or finish
    all_open_cells = []
    for y in range(height):
        for x in range(width):
            if maze[y][x] == 0:
                if not (x == 0 and y == 0) and not (x == 49 and y == 49):
                    all_open_cells.append((x, y))
    
    random.shuffle(all_open_cells)
    
    # -- Add Teleporters (2) --
    # Constraint: "Number may be higher than 2".
    # Constraint: "One correct path".
    # Strategy: Place teleporters ONLY in dead ends (cells not in solution_path_set).
    # This creates traps. If you take a teleporter, you are definitely off the main path.
    
    dead_end_cells = [c for c in all_open_cells if c not in solution_path_set]
    
    # We add pairs of teleporters. Let's add 3 to 5 pairs (6 to 10 endpoints) for difficulty.
    num_teleporter_pairs = random.randint(3, 5)
    teleporters_placed = 0
    
    while teleporters_placed < num_teleporter_pairs * 2 and len(dead_end_cells) >= 2:
        # Pop two random dead end locations
        t1 = dead_end_cells.pop()
        t2 = dead_end_cells.pop()
        
        maze[t1[1]][t1[0]] = 2
        maze[t2[1]][t2[0]] = 2
        teleporters_placed += 2
        
        # Remove from general pool so we don't overwrite with quicksand
        if t1 in all_open_cells: all_open_cells.remove(t1)
        if t2 in all_open_cells: all_open_cells.remove(t2)

    # -- Add Quicksand (3) --
    # Quicksand can be on the main path or dead ends. It just increases cost.
    # We'll make about 10% of the remaining open space quicksand.
    
    num_quicksand = int(len(all_open_cells) * 0.10) 
    
    for _ in range(num_quicksand):
        if not all_open_cells: break
        qx, qy = all_open_cells.pop()
        maze[qy][qx] = 3

    return maze

def visualize(maze):
    """
    Visualizes the maze by printing each row on a new line.
    """
    print(f"Maze Dimensions: {len(maze[0])}x{len(maze)}")
    print("Legend: 0=Path, 1=Wall, 2=Teleporter, 3=Quicksand")
    print("-" * 30)
    for row in maze:
        print(row)

# ---------------------------------------------------------
# Usage Example
# ---------------------------------------------------------
if __name__ == "__main__":
    # Generate the maze
    t1 = time.perf_counter()
    maze = generate_maze()
    t2 = time.perf_counter()
    print(t2-t1)
    
    # Visualize it
    # visualize(maze)