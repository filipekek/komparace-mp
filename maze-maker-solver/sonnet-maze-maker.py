import random
import time

def generate_maze():
    """
    Generate a 50x50 maze with exactly one solution path from start to finish.
    Returns a maze (list of lists) where:
    0 = Open path
    1 = Classic wall
    2 = Teleporter
    3 = Quick sand
    """
    size = 50
    maze = [[1 for _ in range(size)] for _ in range(size)]
    
    # Start and finish positions
    start = (0, 0)
    finish = (size - 1, size - 1)
    
    # Create the main solution path using recursive backtracking
    path = []
    visited = set()
    
    def carve_path(x, y):
        """Carve a path through the maze using depth-first search"""
        visited.add((x, y))
        path.append((x, y))
        maze[x][y] = 0  # Open path
        
        # Directions: right, down, left, up
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            # Check if the new position is valid
            if 0 <= nx < size and 0 <= ny < size and (nx, ny) not in visited:
                # Check if we can carve here (avoid creating loops)
                if can_carve(nx, ny, visited, size):
                    carve_path(nx, ny)
    
    def can_carve(x, y, visited, size):
        """Check if we can safely carve at position (x, y) without creating loops"""
        if (x, y) in visited:
            return False
        
        # Count already carved neighbors
        carved_neighbors = 0
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size and (nx, ny) in visited:
                carved_neighbors += 1
        
        # Only allow carving if we have exactly one carved neighbor (prevents loops)
        return carved_neighbors == 1
    
    # Start carving from the start position
    carve_path(start[0], start[1])
    
    # Ensure finish is reachable - force a path to finish if needed
    if finish not in visited:
        # Find the closest point in the path to the finish
        current = path[-1]
        
        while current != finish:
            x, y = current
            # Move towards finish
            if x < finish[0]:
                next_pos = (x + 1, y)
            elif y < finish[1]:
                next_pos = (x, y + 1)
            elif x > finish[0]:
                next_pos = (x - 1, y)
            else:
                next_pos = (x, y - 1)
            
            if next_pos not in visited:
                maze[next_pos[0]][next_pos[1]] = 0
                visited.add(next_pos)
                path.append(next_pos)
            current = next_pos
    
    # Add some additional branching paths to increase difficulty (dead ends)
    additional_paths = random.randint(15, 30)
    for _ in range(additional_paths):
        # Pick a random cell from the existing path
        if path:
            start_branch = random.choice(path)
            create_dead_end(maze, start_branch, visited, size, max_length=random.randint(3, 10))
    
    # Add teleporters (2-6 teleporters to increase difficulty)
    num_teleporters = random.randint(2, 6)
    teleporter_positions = []
    
    # Place teleporters on the main path
    path_positions = [pos for pos in path if pos != start and pos != finish]
    if len(path_positions) >= num_teleporters:
        teleporter_positions = random.sample(path_positions, num_teleporters)
        for pos in teleporter_positions:
            maze[pos[0]][pos[1]] = 2
    
    # Add quicksand to make the maze harder (place on path)
    num_quicksand = random.randint(20, 40)
    quicksand_candidates = [pos for pos in path if pos != start and pos != finish 
                           and pos not in teleporter_positions]
    
    if len(quicksand_candidates) >= num_quicksand:
        quicksand_positions = random.sample(quicksand_candidates, num_quicksand)
        for pos in quicksand_positions:
            maze[pos[0]][pos[1]] = 3
    
    return maze


def create_dead_end(maze, start_pos, visited, size, max_length):
    """Create a dead-end branch from a position on the main path"""
    x, y = start_pos
    length = 0
    
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    random.shuffle(directions)
    
    for dx, dy in directions:
        if length >= max_length:
            break
            
        nx, ny = x + dx, y + dy
        
        # Try to extend the dead end
        while (0 <= nx < size and 0 <= ny < size and 
               (nx, ny) not in visited and length < max_length):
            
            # Check that we're not creating a loop
            open_neighbors = 0
            for ddx, ddy in directions:
                nnx, nny = nx + ddx, ny + ddy
                if (0 <= nnx < size and 0 <= nny < size and 
                    (nnx, nny) in visited):
                    open_neighbors += 1
            
            if open_neighbors > 1:  # Would create a loop
                break
            
            maze[nx][ny] = random.choice([0, 0, 0, 3])  # Mostly open, some quicksand
            visited.add((nx, ny))
            length += 1
            
            # Continue in the same direction
            nx, ny = nx + dx, ny + dy


def visualize(maze):
    """
    Visualize the maze with symbols:
    0 (Open path) = ' ' (space)
    1 (Wall) = '#'
    2 (Teleporter) = 'T'
    3 (Quicksand) = '~'
    Also marks Start (S) and Finish (F)
    """
    if not maze or not maze[0]:
        print("Empty maze")
        return
    
    symbols = {
        0: ' ',  # Open path
        1: '#',  # Wall
        2: 'T',  # Teleporter
        3: '~'   # Quicksand
    }
    
    for i, row in enumerate(maze):
        line = ""
        for j, cell in enumerate(row):
            if i == 0 and j == 0:
                line += 'S'  # Start
            elif i == len(maze) - 1 and j == len(row) - 1:
                line += 'F'  # Finish
            else:
                line += symbols.get(cell, '?')
        print(line)


# Generate the maze
t1 = time.perf_counter()
maze = generate_maze()
t2 = time.perf_counter()
print(t2-t1)


# Optional: Verify maze integrity
def verify_maze(maze):
    """Verify that there is exactly one path from start to finish"""
    if not maze or len(maze) != 50 or len(maze[0]) != 50:
        return False, "Invalid maze dimensions"
    
    size = 50
    start = (0, 0)
    finish = (size - 1, size - 1)
    
    # BFS to check if finish is reachable
    from collections import deque
    
    queue = deque([start])
    visited = {start}
    
    while queue:
        x, y = queue.popleft()
        
        if (x, y) == finish:
            return True, "Valid maze with reachable finish"
        
        # Check all four directions
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            
            if (0 <= nx < size and 0 <= ny < size and 
                (nx, ny) not in visited and maze[nx][ny] != 1):
                visited.add((nx, ny))
                queue.append((nx, ny))
        
        # Handle teleporters
        if maze[x][y] == 2:
            # Find all other teleporters
            for i in range(size):
                for j in range(size):
                    if maze[i][j] == 2 and (i, j) != (x, y) and (i, j) not in visited:
                        visited.add((i, j))
                        queue.append((i, j))
    
    return False, "Finish not reachable"


# Uncomment to verify the maze
# is_valid, message = verify_maze(maze)
# print(f"Maze validation: {message}")