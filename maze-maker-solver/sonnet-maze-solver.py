def solve_maze(maze):
    """
    Solves a maze and visualizes the solution with move counts.
    
    Args:
        maze: List of lists where:
              0 = Open path
              1 = Wall
              2 = Teleporter
              3 = Quicksand (costs 2 moves instead of 1)
    
    Returns:
        None (prints the solved maze and move counts)
    """
    if not maze or not maze[0]:
        print("Error: Empty maze provided")
        return
    
    rows = len(maze)
    cols = len(maze[0])
    
    # Validate maze structure
    for row in maze:
        if len(row) != cols:
            print("Error: Maze rows have inconsistent lengths")
            return
    
    # Check if start and end are valid
    if maze[0][0] == 1 or maze[rows-1][cols-1] == 1:
        print("Error: Start or end position is a wall")
        return
    
    # Find all teleporters
    teleporters = []
    for r in range(rows):
        for c in range(cols):
            if maze[r][c] == 2:
                teleporters.append((r, c))
    
    # Track visited cells and the path
    visited = set()
    path = []
    total_moves = 0
    correct_path_moves = 0
    
    # Color map: None = unvisited, 'red' = dead end, 'green' = correct path
    color_map = [[None for _ in range(cols)] for _ in range(rows)]
    
    def get_move_cost(cell_type):
        """Returns the move cost for a cell type"""
        if cell_type == 3:  # Quicksand
            return 2
        return 1
    
    def is_valid(r, c, visited_set):
        """Check if position is valid and can be visited"""
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return False
        if (r, c) in visited_set:
            return False
        if maze[r][c] == 1:  # Wall
            return False
        return True
    
    def get_teleport_destinations(r, c):
        """Get possible teleport destinations from current teleporter"""
        destinations = []
        for tr, tc in teleporters:
            if (tr, tc) != (r, c) and (tr, tc) not in visited:
                destinations.append((tr, tc))
        return destinations
    
    def dfs(r, c, current_moves):
        """Depth-first search to find path from (r,c) to end"""
        nonlocal total_moves
        
        # Mark as visited
        visited.add((r, c))
        path.append((r, c))
        
        # Add move cost
        move_cost = get_move_cost(maze[r][c])
        current_moves += move_cost
        total_moves += move_cost
        
        # Check if we reached the end
        if r == rows - 1 and c == cols - 1:
            return True, current_moves
        
        # Explore neighbors (up, down, left, right)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        next_positions = []
        
        # Add regular moves
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if is_valid(nr, nc, visited):
                next_positions.append((nr, nc))
        
        # Add teleport moves if current cell is a teleporter
        if maze[r][c] == 2:
            teleport_dests = get_teleport_destinations(r, c)
            next_positions.extend(teleport_dests)
        
        # Try each next position
        for nr, nc in next_positions:
            found, final_moves = dfs(nr, nc, current_moves)
            if found:
                return True, final_moves
        
        # Dead end - backtrack
        visited.remove((r, c))
        path.pop()
        color_map[r][c] = 'red'
        
        return False, 0
    
    # Solve the maze
    found, correct_path_moves = dfs(0, 0, 0)
    
    if not found:
        print("No solution found for this maze")
        return
    
    # Color the correct path green
    for r, c in path:
        color_map[r][c] = 'green'
    
    # Print the solved maze
    print("\nSolved Maze:")
    print("=" * (cols * 4 + 1))
    
    # Symbol mapping
    symbols = {
        0: ' ',
        1: '█',
        2: 'T',
        3: '~'
    }
    
    # Color codes for terminal
    colors = {
        'green': '\033[92m',
        'red': '\033[91m',
        'reset': '\033[0m'
    }
    
    for r in range(rows):
        row_str = ""
        for c in range(cols):
            cell_value = maze[r][c]
            symbol = symbols.get(cell_value, '?')
            
            # Apply coloring
            if color_map[r][c] == 'green':
                if r == 0 and c == 0:
                    cell_str = f"{colors['green']}[S]{colors['reset']}"
                elif r == rows - 1 and c == cols - 1:
                    cell_str = f"{colors['green']}[E]{colors['reset']}"
                else:
                    cell_str = f"{colors['green']} {symbol} {colors['reset']}"
            elif color_map[r][c] == 'red':
                cell_str = f"{colors['red']} {symbol} {colors['reset']}"
            else:
                if cell_value == 1:
                    cell_str = f" {symbol} "
                else:
                    cell_str = f" {symbol} "
            
            row_str += cell_str + " "
        
        print(row_str)
    
    print("=" * (cols * 4 + 1))
    print(f"\nCorrect path moves: {correct_path_moves}")
    print(f"Total moves (including dead ends): {total_moves}")
    print(f"\nLegend:")
    print(f"  {colors['green']}[S]{colors['reset']} = Start")
    print(f"  {colors['green']}[E]{colors['reset']} = End")
    print(f"  {colors['green']}Green path{colors['reset']} = Correct solution")
    print(f"  {colors['red']}Red path{colors['reset']} = Dead ends explored")
    print(f"  █ = Wall")
    print(f"  T = Teleporter")
    print(f"  ~ = Quicksand (costs 2 moves)")
    print(f"    = Open path")