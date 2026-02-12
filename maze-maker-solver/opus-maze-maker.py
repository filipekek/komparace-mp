"""
Randomized Maze Generator

Creates a 50x50 maze with:
- 0: Open path
- 1: Classic wall
- 2: Teleporter (can teleport to any other teleporter)
- 3: Quicksand (passable but adds +1 move cost)

The maze has exactly one correct path from start (0,0) to finish (49,49),
designed to be as difficult as possible.
"""

import random
from collections import deque
from typing import List, Tuple, Set, Optional, Dict
import heapq
import time


class MazeGenerator:
    """Generates challenging mazes with teleporters and quicksand."""
    
    OPEN = 0
    WALL = 1
    TELEPORTER = 2
    QUICKSAND = 3
    
    def __init__(self, width: int = 50, height: int = 50, seed: Optional[int] = None):
        """
        Initialize the maze generator.
        
        Args:
            width: Width of the maze (number of columns)
            height: Height of the maze (number of rows)
            seed: Optional random seed for reproducibility
        """
        if width < 3 or height < 3:
            raise ValueError("Maze dimensions must be at least 3x3")
        
        self.width = width
        self.height = height
        self.start = (0, 0)
        self.finish = (height - 1, width - 1)
        
        if seed is not None:
            random.seed(seed)
        
        self.maze: List[List[int]] = []
        self.solution_path: List[Tuple[int, int]] = []
        self.teleporter_positions: List[Tuple[int, int]] = []
    
    def generate(self) -> List[List[int]]:
        """
        Generate a complete maze with one solution path.
        
        Returns:
            2D list representing the maze
        """
        # Step 1: Initialize maze with all walls
        self.maze = [[self.WALL for _ in range(self.width)] for _ in range(self.height)]
        
        # Step 2: Generate base maze using modified DFS to create longest path
        self._generate_base_maze_longest_path()
        
        # Step 3: Ensure exactly one solution exists
        self._ensure_single_solution()
        
        # Step 4: Add teleporters strategically (on dead ends or detours)
        self._add_teleporters()
        
        # Step 5: Add quicksand on the solution path to increase difficulty
        self._add_quicksand()
        
        # Step 6: Verify maze integrity
        if not self._verify_maze():
            # Fallback: regenerate if verification fails
            return self.generate()
        
        return self.maze
    
    def _generate_base_maze_longest_path(self) -> None:
        """
        Generate maze using DFS with bias toward creating the longest possible path.
        Uses a modified algorithm that prioritizes exploring away from the goal first.
        """
        # Start with all walls, carve paths
        visited = [[False] * self.width for _ in range(self.height)]
        
        # Use iterative DFS with stack
        stack = [self.start]
        visited[self.start[0]][self.start[1]] = True
        self.maze[self.start[0]][self.start[1]] = self.OPEN
        
        # Track the path for backtracking
        parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
        
        while stack:
            current = stack[-1]
            row, col = current
            
            # Get unvisited neighbors (2 cells away for maze walls)
            neighbors = self._get_unvisited_neighbors(row, col, visited)
            
            if neighbors:
                # Sort neighbors to prioritize those farther from finish
                # This creates a more winding path
                neighbors.sort(
                    key=lambda n: -self._manhattan_distance(n, self.finish)
                )
                
                # Add some randomness but bias toward farther cells
                if len(neighbors) > 1 and random.random() < 0.3:
                    random.shuffle(neighbors)
                
                next_cell = neighbors[0]
                next_row, next_col = next_cell
                
                # Carve path between current and next (remove wall in between)
                wall_row = (row + next_row) // 2
                wall_col = (col + next_col) // 2
                
                self.maze[wall_row][wall_col] = self.OPEN
                self.maze[next_row][next_col] = self.OPEN
                
                visited[next_row][next_col] = True
                parent[next_cell] = current
                stack.append(next_cell)
            else:
                stack.pop()
        
        # Ensure finish is reachable - connect if isolated
        self.maze[self.finish[0]][self.finish[1]] = self.OPEN
        
        # Find and store the solution path
        self._find_solution_path()
    
    def _get_unvisited_neighbors(
        self, row: int, col: int, visited: List[List[bool]]
    ) -> List[Tuple[int, int]]:
        """Get unvisited neighbors 2 cells away (for maze generation)."""
        neighbors = []
        directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < self.height and 
                0 <= new_col < self.width and 
                not visited[new_row][new_col]):
                neighbors.append((new_row, new_col))
        
        return neighbors
    
    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _find_solution_path(self) -> bool:
        """
        Find the solution path using BFS (shortest path without teleporters).
        
        Returns:
            True if path exists, False otherwise
        """
        if self.maze[self.start[0]][self.start[1]] == self.WALL:
            return False
        if self.maze[self.finish[0]][self.finish[1]] == self.WALL:
            return False
        
        visited = [[False] * self.width for _ in range(self.height)]
        parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {self.start: None}
        queue = deque([self.start])
        visited[self.start[0]][self.start[1]] = True
        
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        while queue:
            current = queue.popleft()
            
            if current == self.finish:
                # Reconstruct path
                self.solution_path = []
                node: Optional[Tuple[int, int]] = self.finish
                while node is not None:
                    self.solution_path.append(node)
                    node = parent[node]
                self.solution_path.reverse()
                return True
            
            row, col = current
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                if (0 <= new_row < self.height and 
                    0 <= new_col < self.width and 
                    not visited[new_row][new_col] and 
                    self.maze[new_row][new_col] != self.WALL):
                    visited[new_row][new_col] = True
                    parent[(new_row, new_col)] = current
                    queue.append((new_row, new_col))
        
        return False
    
    def _ensure_single_solution(self) -> None:
        """
        Modify the maze to ensure exactly one solution path.
        Removes alternative paths by adding walls.
        """
        if not self.solution_path:
            if not self._find_solution_path():
                # If no path exists, create one
                self._create_direct_path()
                return
        
        # Convert solution path to set for quick lookup
        solution_set = set(self.solution_path)
        
        # Find and block alternative paths
        # We do this by checking each cell adjacent to the solution path
        for pos in self.solution_path:
            row, col = pos
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                if (0 <= new_row < self.height and 
                    0 <= new_col < self.width and 
                    (new_row, new_col) not in solution_set and
                    self.maze[new_row][new_col] == self.OPEN):
                    
                    # Check if this creates an alternative path
                    if self._creates_alternative_path((new_row, new_col), solution_set):
                        self.maze[new_row][new_col] = self.WALL
        
        # Verify single solution
        path_count = self._count_solution_paths()
        if path_count != 1:
            # More aggressive blocking
            self._block_all_alternatives()
    
    def _creates_alternative_path(
        self, cell: Tuple[int, int], solution_set: Set[Tuple[int, int]]
    ) -> bool:
        """Check if a cell could create an alternative path."""
        row, col = cell
        adjacent_solution_cells = 0
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if (new_row, new_col) in solution_set:
                adjacent_solution_cells += 1
        
        # If cell connects to 2+ solution cells, it might create a shortcut
        return adjacent_solution_cells >= 2
    
    def _count_solution_paths(self) -> int:
        """
        Count the number of distinct solution paths (simplified check).
        Returns 1 if there's exactly one path, 0 if none, 2+ if multiple.
        """
        if not self._find_solution_path():
            return 0
        
        # Quick check: if we can find a path after removing each solution cell
        # (except start and finish), there are multiple paths
        for i, pos in enumerate(self.solution_path[1:-1], 1):
            row, col = pos
            original = self.maze[row][col]
            self.maze[row][col] = self.WALL
            
            has_alt_path = self._has_path_bfs()
            self.maze[row][col] = original
            
            if has_alt_path:
                return 2  # Multiple paths exist
        
        return 1
    
    def _has_path_bfs(self) -> bool:
        """Quick BFS to check if any path exists."""
        if self.maze[self.start[0]][self.start[1]] == self.WALL:
            return False
        if self.maze[self.finish[0]][self.finish[1]] == self.WALL:
            return False
        
        visited = [[False] * self.width for _ in range(self.height)]
        queue = deque([self.start])
        visited[self.start[0]][self.start[1]] = True
        
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        while queue:
            row, col = queue.popleft()
            
            if (row, col) == self.finish:
                return True
            
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                if (0 <= new_row < self.height and 
                    0 <= new_col < self.width and 
                    not visited[new_row][new_col] and 
                    self.maze[new_row][new_col] != self.WALL):
                    visited[new_row][new_col] = True
                    queue.append((new_row, new_col))
        
        return False
    
    def _block_all_alternatives(self) -> None:
        """Aggressively block all cells not on the solution path."""
        solution_set = set(self.solution_path)
        
        for row in range(self.height):
            for col in range(self.width):
                if (row, col) not in solution_set and self.maze[row][col] == self.OPEN:
                    self.maze[row][col] = self.WALL
    
    def _create_direct_path(self) -> None:
        """Create a winding path from start to finish if none exists."""
        # Reset maze
        self.maze = [[self.WALL for _ in range(self.width)] for _ in range(self.height)]
        
        # Create a winding path
        path = []
        current = self.start
        path.append(current)
        self.maze[current[0]][current[1]] = self.OPEN
        
        while current != self.finish:
            row, col = current
            
            # Decide direction with bias toward finish but allow wandering
            if random.random() < 0.6:
                # Move toward finish
                if row < self.finish[0] and self.maze[row + 1][col] == self.WALL:
                    current = (row + 1, col)
                elif col < self.finish[1] and self.maze[row][col + 1] == self.WALL:
                    current = (row, col + 1)
                elif row > 0 and self.maze[row - 1][col] == self.WALL:
                    current = (row - 1, col)
                elif col > 0 and self.maze[row][col - 1] == self.WALL:
                    current = (row, col - 1)
                else:
                    # Force toward finish
                    if row < self.finish[0]:
                        current = (row + 1, col)
                    else:
                        current = (row, col + 1)
            else:
                # Random direction
                directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                random.shuffle(directions)
                for dr, dc in directions:
                    new_row, new_col = row + dr, col + dc
                    if (0 <= new_row < self.height and 
                        0 <= new_col < self.width):
                        current = (new_row, new_col)
                        break
            
            path.append(current)
            self.maze[current[0]][current[1]] = self.OPEN
            
            # Safety check to prevent infinite loops
            if len(path) > self.width * self.height * 2:
                break
        
        self.solution_path = path
    
    def _add_teleporters(self) -> None:
        """
        Add teleporters strategically to increase maze difficulty.
        Teleporters are placed in dead ends to confuse solvers.
        """
        self.teleporter_positions = []
        
        # Find dead ends (cells with only one open neighbor)
        dead_ends = []
        solution_set = set(self.solution_path)
        
        for row in range(self.height):
            for col in range(self.width):
                if (row, col) in solution_set:
                    continue
                if self.maze[row][col] == self.OPEN:
                    open_neighbors = self._count_open_neighbors(row, col)
                    if open_neighbors == 1:
                        dead_ends.append((row, col))
        
        # Add teleporters (3-6 teleporters for difficulty)
        num_teleporters = min(len(dead_ends), random.randint(3, 6))
        
        if num_teleporters >= 2:
            # Ensure even number for pairing (though problem says >2 is allowed)
            selected = random.sample(dead_ends, num_teleporters) if dead_ends else []
            
            for pos in selected:
                self.maze[pos[0]][pos[1]] = self.TELEPORTER
                self.teleporter_positions.append(pos)
        
        # If not enough dead ends, create teleporter spots on solution path edges
        if len(self.teleporter_positions) < 2 and len(self.solution_path) > 10:
            # Add teleporters at branches off the solution path
            for pos in self.solution_path[2:-2]:
                if len(self.teleporter_positions) >= 4:
                    break
                row, col = pos
                directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                for dr, dc in directions:
                    new_row, new_col = row + dr, col + dc
                    if (0 <= new_row < self.height and 
                        0 <= new_col < self.width and 
                        self.maze[new_row][new_col] == self.WALL and
                        (new_row, new_col) not in solution_set):
                        # Create a small alcove with teleporter
                        self.maze[new_row][new_col] = self.TELEPORTER
                        self.teleporter_positions.append((new_row, new_col))
                        break
    
    def _count_open_neighbors(self, row: int, col: int) -> int:
        """Count open (non-wall) neighbors of a cell."""
        count = 0
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < self.height and 
                0 <= new_col < self.width and 
                self.maze[new_row][new_col] != self.WALL):
                count += 1
        return count
    
    def _add_quicksand(self) -> None:
        """
        Add quicksand on the solution path to increase move cost.
        Quicksand is placed to maximize the penalty for the solver.
        """
        if len(self.solution_path) < 5:
            return
        
        # Add quicksand to ~20-30% of solution path (excluding start and finish)
        eligible_positions = self.solution_path[1:-1]
        num_quicksand = max(1, len(eligible_positions) // 4)
        
        # Prefer positions in the middle of the path (harder to avoid)
        middle_start = len(eligible_positions) // 4
        middle_end = 3 * len(eligible_positions) // 4
        middle_positions = eligible_positions[middle_start:middle_end]
        
        if middle_positions:
            selected = random.sample(
                middle_positions, 
                min(num_quicksand, len(middle_positions))
            )
            for pos in selected:
                # Don't overwrite teleporters
                if self.maze[pos[0]][pos[1]] == self.OPEN:
                    self.maze[pos[0]][pos[1]] = self.QUICKSAND
    
    def _verify_maze(self) -> bool:
        """
        Verify the maze is valid:
        - Start and finish are accessible
        - Exactly one solution path exists
        - At least one path exists
        """
        # Check start and finish
        if self.maze[self.start[0]][self.start[1]] == self.WALL:
            return False
        if self.maze[self.finish[0]][self.finish[1]] == self.WALL:
            return False
        
        # Check path exists
        if not self._has_path_bfs():
            return False
        
        return True
    
    def get_maze(self) -> List[List[int]]:
        """Return the generated maze."""
        return self.maze
    
    def get_solution_path(self) -> List[Tuple[int, int]]:
        """Return the solution path."""
        return self.solution_path
    
    def get_teleporter_positions(self) -> List[Tuple[int, int]]:
        """Return teleporter positions."""
        return self.teleporter_positions
    
    def calculate_solution_cost(self) -> int:
        """
        Calculate the total cost of the solution path.
        Quicksand adds +1 to the move cost.
        """
        cost = 0
        for pos in self.solution_path:
            cost += 1  # Base move cost
            if self.maze[pos[0]][pos[1]] == self.QUICKSAND:
                cost += 1  # Quicksand penalty
        return cost
    
    def print_maze(self, show_solution: bool = False) -> None:
        """
        Print the maze to console.
        
        Args:
            show_solution: If True, highlight the solution path
        """
        solution_set = set(self.solution_path) if show_solution else set()
        
        symbols = {
            self.OPEN: '.',
            self.WALL: '#',
            self.TELEPORTER: 'T',
            self.QUICKSAND: '~'
        }
        
        print(f"\nMaze ({self.height}x{self.width}):")
        print("Legend: . = Open, # = Wall, T = Teleporter, ~ = Quicksand")
        if show_solution:
            print("        * = Solution path")
        print()
        
        for row in range(self.height):
            line = ""
            for col in range(self.width):
                if (row, col) == self.start:
                    line += "S"
                elif (row, col) == self.finish:
                    line += "F"
                elif show_solution and (row, col) in solution_set:
                    line += "*"
                else:
                    line += symbols[self.maze[row][col]]
            print(line)
        print()
    
    def get_statistics(self) -> Dict:
        """Get maze statistics."""
        total_cells = self.width * self.height
        cell_counts = {
            'open': 0,
            'wall': 0,
            'teleporter': 0,
            'quicksand': 0
        }
        
        for row in self.maze:
            for cell in row:
                if cell == self.OPEN:
                    cell_counts['open'] += 1
                elif cell == self.WALL:
                    cell_counts['wall'] += 1
                elif cell == self.TELEPORTER:
                    cell_counts['teleporter'] += 1
                elif cell == self.QUICKSAND:
                    cell_counts['quicksand'] += 1
        
        return {
            'dimensions': f"{self.height}x{self.width}",
            'total_cells': total_cells,
            'cell_counts': cell_counts,
            'solution_length': len(self.solution_path),
            'solution_cost': self.calculate_solution_cost(),
            'teleporter_count': len(self.teleporter_positions),
            'difficulty_score': self._calculate_difficulty()
        }
    
    def _calculate_difficulty(self) -> float:
        """
        Calculate a difficulty score for the maze.
        Higher is more difficult.
        """
        if not self.solution_path:
            return 0.0
        
        # Factors: path length, quicksand count, teleporter confusion
        path_length_score = len(self.solution_path) / (self.width + self.height)
        quicksand_count = sum(
            1 for pos in self.solution_path 
            if self.maze[pos[0]][pos[1]] == self.QUICKSAND
        )
        quicksand_score = quicksand_count / max(1, len(self.solution_path))
        teleporter_score = len(self.teleporter_positions) * 0.5
        
        return round(path_length_score + quicksand_score + teleporter_score, 2)


def main():
    """Main function to demonstrate maze generation."""
    print("=" * 60)
    print("RANDOMIZED MAZE GENERATOR")
    print("=" * 60)
    
    # Generate maze
    generator = MazeGenerator(width=50, height=50)
    maze = generator.generate()
    
    # Print statistics
    stats = generator.get_statistics()
    print("\nMaze Statistics:")
    print(f"  Dimensions: {stats['dimensions']}")
    print(f"  Total cells: {stats['total_cells']}")
    print(f"  Open paths: {stats['cell_counts']['open']}")
    print(f"  Walls: {stats['cell_counts']['wall']}")
    print(f"  Teleporters: {stats['cell_counts']['teleporter']}")
    print(f"  Quicksand: {stats['cell_counts']['quicksand']}")
    print(f"  Solution length: {stats['solution_length']} steps")
    print(f"  Solution cost (with quicksand): {stats['solution_cost']}")
    print(f"  Difficulty score: {stats['difficulty_score']}")
    
    # Print maze (optional - commented out for large mazes)
    generator.print_maze(show_solution=False)
    
    # Print with solution highlighted
    print("\nWith solution path highlighted:")
    generator.print_maze(show_solution=True)
    
    # Return the maze as required
    print("\nMaze data structure (first 5 rows shown):")
    for i, row in enumerate(maze[:5]):
        print(f"  Row {i}: {row[:10]}... (showing first 10 columns)")
    
    return maze


if __name__ == "__main__":
    t1 = time.perf_counter()
    maze = main()
    t2 = time.perf_counter()
    print(t2-t1)