import random
from typing import List, Tuple, Set

class MazeGenerator:
    # Constants for grid values
    PATH = 0
    WALL = 1
    TELEPORTER = 2
    QUICKSAND = 3
    
    # Grid dimensions
    ROWS = 50
    COLS = 50

    def __init__(self):
        self.grid = [[self.WALL for _ in range(self.COLS)] for _ in range(self.ROWS)]
        self.teleporter_placed = False
    
    def _is_valid(self, r: int, c: int) -> bool:
        """Check if coordinates are within the grid bounds."""
        return 0 <= r < self.ROWS and 0 <= c < self.COLS

    def _get_unvisited_neighbors(self, r: int, c: int, visited: Set[Tuple[int, int]]) -> List[Tuple[int, int, int, int]]:
        """
        Returns a list of unvisited neighbors (distance 2).
        Format: (neighbor_row, neighbor_col, wall_between_row, wall_between_col)
        """
        directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]
        neighbors = []
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.ROWS - 1 and 0 <= nc < self.COLS - 1:
                if (nr, nc) not in visited:
                    # Calculate the wall position between current and neighbor
                    wr, wc = r + (dr // 2), c + (dc // 2)
                    neighbors.append((nr, nc, wr, wc))
        return neighbors

    def _add_quicksand_obstacles(self, density: float = 0.15):
        """
        Randomly converts open paths (0) to Quicksand (3).
        Density determines the percentage of path converted.
        """
        path_cells = []
        for r in range(self.ROWS):
            for c in range(self.COLS):
                # We can place quicksand on 0, but avoid Start and End for clarity
                if self.grid[r][c] == self.PATH:
                    if (r, c) != (0, 0) and (r, c) != (self.ROWS-1, self.COLS-1):
                        path_cells.append((r, c))
        
        # Randomly select cells to turn into quicksand
        num_quicksand = int(len(path_cells) * density)
        if num_quicksand > 0:
            chosen = random.sample(path_cells, num_quicksand)
            for r, c in chosen:
                self.grid[r][c] = self.QUICKSAND

    def generate(self) -> List[List[int]]:
        """
        Generates the 50x50 maze.
        Returns: List of lists (integers).
        """
        # Reset grid
        self.grid = [[self.WALL for _ in range(self.COLS)] for _ in range(self.ROWS)]
        self.teleporter_placed = False
        
        # Using iterative stack to prevent RecursionError on large grids
        # Starting at 0,0
        start_node = (0, 0)
        self.grid[0][0] = self.PATH
        
        stack = [start_node]
        visited = {start_node}
        
        # We only generate 'rooms' on even indices to ensure walls exist between them
        # Valid rooms are (0,0), (0,2), (2,2)... (48,48)
        
        while stack:
            current_r, current_c = stack[-1]
            
            # 1. Check for valid physical neighbors (up, down, left, right)
            neighbors = self._get_unvisited_neighbors(current_r, current_c, visited)
            
            # 2. Logic for Teleporter Injection (The "Bridge" Strategy)
            # To maintain ONE path, we can jump to a random unvisited spot 
            # effectively acting as a bridge instead of carving a wall.
            if not self.teleporter_placed and len(stack) > 10:
                # 2% chance to spawn a teleporter if we are deep enough in the maze
                if random.random() < 0.02:
                    # Find all unvisited valid "room" candidates
                    candidates = []
                    for r in range(0, self.ROWS - 1, 2):
                        for c in range(0, self.COLS - 1, 2):
                            if (r, c) not in visited:
                                candidates.append((r, c))
                    
                    if candidates:
                        target_r, target_c = random.choice(candidates)
                        
                        # Place Teleporter Pair
                        self.grid[current_r][current_c] = self.TELEPORTER
                        self.grid[target_r][target_c] = self.TELEPORTER
                        self.teleporter_placed = True
                        
                        # "Jump" the DFS to the new target
                        visited.add((target_r, target_c))
                        stack.append((target_r, target_c))
                        continue

            if neighbors:
                # Choose a random neighbor
                next_r, next_c, wall_r, wall_c = random.choice(neighbors)
                
                # Carve the path
                self.grid[wall_r][wall_c] = self.PATH # Remove wall between
                self.grid[next_r][next_c] = self.PATH # Mark next cell open
                
                visited.add((next_r, next_c))
                stack.append((next_r, next_c))
            else:
                # Backtrack
                stack.pop()

        # 3. Handle Exit
        # Because we used step=2, the DFS reaches (48, 48). 
        # We must manually open the exit at (49, 49).
        self.grid[self.ROWS-1][self.COLS-1] = self.PATH
        
        # Connect the exit to the maze (usually connecting 48,49 or 49,48)
        # We check which neighbor (48,49) or (49,48) is closest to a carved path
        if self.grid[self.ROWS-2][self.COLS-1] != self.WALL:
             pass # Already accessible vertically
        elif self.grid[self.ROWS-1][self.COLS-2] != self.WALL:
             pass # Already accessible horizontally
        else:
            # Force connection from (48, 48) -> (49, 49)
            # Since (48,48) is the last visited 'room', we open the walls to reach (49,49)
            self.grid[self.ROWS-2][self.COLS-1] = self.PATH # Open (48, 49)
            # Ensure (48,48) was visited/open (it should be, but for safety)
            if self.grid[self.ROWS-2][self.COLS-2] == self.WALL:
                self.grid[self.ROWS-2][self.COLS-2] = self.PATH

        # 4. Add Quicksand
        self._add_quicksand_obstacles(density=0.10)
        
        # Ensure Start and End are strictly 0 (Open) to avoid unfair starts/ends
        self.grid[0][0] = self.PATH
        self.grid[self.ROWS-1][self.COLS-1] = self.PATH

        return self.grid

# ==========================================
# Example Usage & Visualization
# ==========================================
if __name__ == "__main__":
    generator = MazeGenerator()
    maze = generator.generate()

    print(f"Maze Generated: {len(maze)}x{len(maze[0])}")
    print("Legend: 0=Path, 1=Wall, 2=Teleporter, 3=Quicksand")
    print("-" * 30)

    # Simple visualizer for the console
    # Symbols: ' ' = Path, '#' = Wall, 'T' = Teleporter, '~' = Quicksand
    chars = {0: '  ', 1: '##', 2: 'TP', 3: '~~'}
    
    for row in maze:
        line = "".join([chars[cell] for cell in row])
        print(line)

    print("-" * 30)
    print(f"Start: (0,0) | End: (49,49)")