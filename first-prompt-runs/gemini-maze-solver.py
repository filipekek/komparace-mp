import heapq
import sys

class MazeSolver:
    # ANSI Color Codes for Console Output
    COLOR_RESET = "\033[0m"
    COLOR_GREEN = "\033[92m"  # For the correct path
    COLOR_RED = "\033[91m"    # For visited dead ends/explored nodes
    COLOR_BLUE = "\033[94m"   # For Teleporters
    COLOR_YELLOW = "\033[93m" # For Quick Sand
    
    # Legend for internal logic
    TYPE_OPEN = 0
    TYPE_WALL = 1
    TYPE_TELEPORT = 2
    TYPE_QUICKSAND = 3

    def __init__(self, maze):
        self.maze = maze
        self.rows = len(maze)
        self.cols = len(maze[0]) if self.rows > 0 else 0
        self.teleporters = self._find_teleporters()
        self.path = []
        self.visited_cells = set()
        self.move_count = 0

    def _find_teleporters(self):
        """Pre-scans the maze to find all teleporter coordinates."""
        locs = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.maze[r][c] == self.TYPE_TELEPORT:
                    locs.append((r, c))
        return locs

    def _is_valid(self, r, c):
        """Checks bounds and wall collisions."""
        return 0 <= r < self.rows and 0 <= c < self.cols and self.maze[r][c] != self.TYPE_WALL

    def _get_weight(self, r, c):
        """Determines the movement cost to enter a specific cell."""
        cell_type = self.maze[r][c]
        if cell_type == self.TYPE_QUICKSAND:
            return 2 # 1 move + 1 extra
        return 1 # Standard move cost

    def solve(self):
        """
        Executes Dijkstra's Algorithm to find the shortest path.
        Returns True if solvable, False otherwise.
        """
        # Edge Case: Empty maze
        if self.rows == 0 or self.cols == 0:
            print("Error: Empty maze provided.")
            return False

        start = (0, 0)
        end = (self.rows - 1, self.cols - 1)

        # Edge Case: Start or End is a wall
        if self.maze[start[0]][start[1]] == self.TYPE_WALL or self.maze[end[0]][end[1]] == self.TYPE_WALL:
            print("Error: Start or Finish point is a wall.")
            return False

        # Priority Queue: stores (current_cost, r, c)
        pq = [(0, start[0], start[1])]
        
        # Dictionary to track minimum cost to reach a node: (r,c) -> cost
        min_costs = {start: 0}
        
        # Dictionary to reconstruct the path: (child_r, child_c) -> (parent_r, parent_c)
        parents = {start: None}

        # Set to track all nodes visited (popped from PQ) for the "Red" coloring
        processed = set()

        while pq:
            cost, r, c = heapq.heappop(pq)
            current_node = (r, c)

            if current_node in processed:
                continue
            processed.add(current_node)

            # Check if we reached the end
            if current_node == end:
                self.move_count = cost
                self.visited_cells = processed
                self._reconstruct_path(parents, end)
                return True

            # Get Neighbors
            neighbors = []
            
            # 1. Physical Neighbors (Up, Down, Left, Right)
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if self._is_valid(nr, nc):
                    # Cost is determined by the tile we are stepping INTO
                    weight = self._get_weight(nr, nc)
                    neighbors.append(((nr, nc), weight))

            # 2. Teleportation Logic
            # If we are currently standing on a teleporter, we can move to any other teleporter
            # We treat the "jump" as instantaneous (cost 0) because the cost was paid entering the tile.
            if self.maze[r][c] == self.TYPE_TELEPORT:
                for tr, tc in self.teleporters:
                    if (tr, tc) != current_node:
                        neighbors.append(((tr, tc), 0))

            # Process Neighbors
            for next_node, weight in neighbors:
                new_cost = cost + weight
                
                # If we found a cheaper way to this neighbor, update it
                if next_node not in min_costs or new_cost < min_costs[next_node]:
                    min_costs[next_node] = new_cost
                    parents[next_node] = current_node
                    heapq.heappush(pq, (new_cost, next_node[0], next_node[1]))

        # If queue empty and end not reached
        self.visited_cells = processed
        return False

    def _reconstruct_path(self, parents, end_node):
        """Backtracks from end to start to build the path list."""
        curr = end_node
        while curr is not None:
            self.path.append(curr)
            curr = parents[curr]
        self.path.reverse() # Start to End

    def print_result(self):
        """Prints the maze with ANSI coloring."""
        print(f"\n{'-'*20} MAZE RESULT {'-'*20}")
        if not self.path and (self.rows > 0 and self.cols > 0):
            print("Maze is unsolvable!")
        
        path_set = set(self.path)

        for r in range(self.rows):
            line = []
            for c in range(self.cols):
                node = (r, c)
                val = self.maze[r][c]
                
                # Determine display character and color
                char = str(val)
                color = self.COLOR_RESET

                if node in path_set:
                    # Logic: Part of the solution path -> GREEN
                    color = self.COLOR_GREEN
                    char = "P" # P for Path
                elif node in self.visited_cells:
                    # Logic: Visited but not path (Dead end/Explored) -> RED
                    color = self.COLOR_RED
                    char = "x" 
                elif val == self.TYPE_WALL:
                    char = "#"
                elif val == self.TYPE_TELEPORT:
                    color = self.COLOR_BLUE
                    char = "T"
                elif val == self.TYPE_QUICKSAND:
                    color = self.COLOR_YELLOW
                    char = "Q"
                else:
                    char = "." # Unvisited open path

                line.append(f"{color} {char} {self.COLOR_RESET}")
            print("".join(line))
        
        print(f"{'-'*53}")
        if self.path:
            print(f"Total Moves: {self.move_count}")
        print(f"{'-'*53}")

# --- User Input Helper ---
def get_user_maze():
    print("Enter the maze row by row. Type numbers separated by spaces.")
    print("Example: 0 1 0 2")
    print("Type 'done' when finished.")
    
    maze = []
    while True:
        line = input(f"Row {len(maze)}: ").strip()
        if line.lower() == 'done':
            break
        try:
            row = [int(x) for x in line.split()]
            if maze and len(row) != len(maze[0]):
                print("Error: Row length mismatch. Please keep the maze rectangular.")
                continue
            maze.append(row)
        except ValueError:
            print("Invalid input. Please enter numbers 0-3 only.")
    return maze

# --- Main Execution ---
# if __name__ == "__main__":
#     # Example 1: Hardcoded for quick testing
#     # 0=Path, 1=Wall, 2=Teleport, 3=Quicksand
#     # Start (0,0), End (Bottom Right)
#     test_maze = [
#         [0, 3, 1, 0, 0],
#         [0, 1, 0, 3, 0],
#         [2, 1, 0, 1, 0],  # Teleporter at (2,0)
#         [0, 3, 0, 2, 0],  # Teleporter at (3,3)
#         [0, 0, 0, 1, 0]   # End at (4,4)
#     ]

#     print("Running Test Case...")
#     solver = MazeSolver(test_maze)
#     if solver.solve():
#         solver.print_result()
#     else:
#         print("No path found.")

#     # Uncomment the lines below to enable manual user input:
#     # user_maze = get_user_maze()
#     # if user_maze:
#     #     solver = MazeSolver(user_maze)
#     #     solver.solve()
#     #     solver.print_result()