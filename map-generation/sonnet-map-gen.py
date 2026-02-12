"""
Map Generator for Video Game World
Generates a 240x160 pixel map with proper biome connectivity rules.
"""

import tkinter as tk
from tkinter import Canvas, Button
import random
import time


class MapGenerator:
    """Generates a diverse pixel map with proper biome connectivity."""
    
    # Map dimensions
    WIDTH = 240
    HEIGHT = 160
    PIXEL_SIZE = 4  # Enlarged pixel size for display
    
    # Biome types
    DEEP_SEA = 0
    SHALLOW_SEA = 1
    RIVER = 2
    BEACH = 3
    PLAINS = 4
    FOREST = 5
    MOUNTAIN = 6
    PEAK = 7
    
    # Color mappings
    COLORS = {
        DEEP_SEA: '#0000CD',      # Dark blue
        SHALLOW_SEA: '#87CEEB',   # Light blue
        RIVER: '#87CEEB',         # Light blue
        BEACH: '#FFD700',         # Yellow
        PLAINS: '#90EE90',        # Light green
        FOREST: '#228B22',        # Dark green
        MOUNTAIN: '#808080',      # Gray
        PEAK: '#FFFFFF'           # White
    }
    
    def __init__(self):
        """Initialize the map grid."""
        self.map = [[self.DEEP_SEA for _ in range(self.WIDTH)] for _ in range(self.HEIGHT)]
    
    def generate_map(self):
        """Generate a new map following all biome connectivity rules."""
        # Reset map to deep sea
        self.map = [[self.DEEP_SEA for _ in range(self.WIDTH)] for _ in range(self.HEIGHT)]
        
        # Step 1: Generate land masses (plains and forests)
        self._generate_land_masses()
        
        # Step 2: Add shallow water around land
        self._generate_shallow_water()
        
        # Step 3: Add beaches between water and land
        self._generate_beaches()
        
        # Step 4: Add mountain ranges on land (away from beaches)
        self._generate_mountains()
        
        # Step 5: Add peaks on mountains
        self._generate_peaks()
        
        # Step 6: Add rivers connecting land to water
        self._generate_rivers()
        
        # Step 7: Ensure all beaches touch water and land
        self._fix_beach_connectivity()
        
        # Step 8: Final validation and cleanup
        self._validate_and_cleanup()
        
        return self.map
    
    def _generate_land_masses(self):
        """Generate land masses using a simple island generation algorithm."""
        num_islands = random.randint(3, 6)
        
        for _ in range(num_islands):
            # Random center point for island
            center_x = random.randint(20, self.WIDTH - 20)
            center_y = random.randint(20, self.HEIGHT - 20)
            
            # Random island size
            radius = random.randint(15, 35)
            
            # Generate island with irregular shape
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    x = center_x + dx
                    y = center_y + dy
                    
                    if 0 <= x < self.WIDTH and 0 <= y < self.HEIGHT:
                        # Distance from center with noise
                        dist = (dx * dx + dy * dy) ** 0.5
                        noise = random.uniform(-5, 5)
                        
                        if dist + noise < radius:
                            # Randomly choose between plains and forest
                            if random.random() < 0.6:
                                self.map[y][x] = self.PLAINS
                            else:
                                self.map[y][x] = self.FOREST
    
    def _generate_shallow_water(self):
        """Generate shallow water around land masses."""
        new_map = [row[:] for row in self.map]
        
        for y in range(self.HEIGHT):
            for x in range(self.WIDTH):
                if self.map[y][x] == self.DEEP_SEA:
                    # Check if adjacent to land
                    if self._has_adjacent_land(x, y):
                        new_map[y][x] = self.SHALLOW_SEA
        
        self.map = new_map
    
    def _generate_beaches(self):
        """Generate beaches between water and land."""
        new_map = [row[:] for row in self.map]
        
        for y in range(self.HEIGHT):
            for x in range(self.WIDTH):
                if self.map[y][x] in [self.PLAINS, self.FOREST]:
                    # Check if adjacent to water
                    if self._has_adjacent_water(x, y):
                        new_map[y][x] = self.BEACH
        
        self.map = new_map
    
    def _generate_mountains(self):
        """Generate mountain ranges on plains or forests away from beaches."""
        num_ranges = random.randint(2, 4)
        
        for _ in range(num_ranges):
            # Find a suitable starting point (plains or forest, not near beach)
            attempts = 0
            while attempts < 100:
                x = random.randint(10, self.WIDTH - 10)
                y = random.randint(10, self.HEIGHT - 10)
                
                if self.map[y][x] in [self.PLAINS, self.FOREST]:
                    if not self._has_beach_nearby(x, y, radius=3):
                        break
                attempts += 1
            else:
                continue  # Couldn't find suitable location
            
            # Generate mountain range
            length = random.randint(15, 30)
            direction = random.choice([(1, 0), (0, 1), (1, 1), (1, -1)])
            
            for i in range(length):
                curr_x = x + direction[0] * i
                curr_y = y + direction[1] * i
                
                # Add some width to the range
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        px = curr_x + dx
                        py = curr_y + dy
                        
                        if 0 <= px < self.WIDTH and 0 <= py < self.HEIGHT:
                            if self.map[py][px] in [self.PLAINS, self.FOREST]:
                                if random.random() < 0.7:
                                    self.map[py][px] = self.MOUNTAIN
    
    def _generate_peaks(self):
        """Generate peaks on mountain ranges."""
        new_map = [row[:] for row in self.map]
        
        for y in range(1, self.HEIGHT - 1):
            for x in range(1, self.WIDTH - 1):
                if self.map[y][x] == self.MOUNTAIN:
                    # Check if surrounded by mountains
                    surrounded = True
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            if self.map[y + dy][x + dx] != self.MOUNTAIN:
                                surrounded = False
                                break
                        if not surrounded:
                            break
                    
                    # Randomly make some surrounded mountains into peaks
                    if surrounded and random.random() < 0.3:
                        new_map[y][x] = self.PEAK
        
        self.map = new_map
    
    def _generate_rivers(self):
        """Generate rivers from land to water."""
        num_rivers = random.randint(2, 5)
        
        for _ in range(num_rivers):
            # Find a starting point on land (away from existing water)
            attempts = 0
            while attempts < 100:
                x = random.randint(5, self.WIDTH - 5)
                y = random.randint(5, self.HEIGHT - 5)
                
                if self.map[y][x] in [self.PLAINS, self.FOREST]:
                    if not self._has_adjacent_water(x, y):
                        break
                attempts += 1
            else:
                continue
            
            # Trace river to nearest water
            river_path = [(x, y)]
            current_x, current_y = x, y
            max_length = 100
            
            for _ in range(max_length):
                # Find nearest water
                nearest_water = self._find_nearest_water(current_x, current_y)
                if nearest_water is None:
                    break
                
                water_x, water_y = nearest_water
                
                # Move one step towards water
                dx = 1 if water_x > current_x else (-1 if water_x < current_x else 0)
                dy = 1 if water_y > current_y else (-1 if water_y < current_y else 0)
                
                # Add some randomness to path
                if random.random() < 0.3:
                    dx, dy = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
                
                current_x += dx
                current_y += dy
                
                # Bounds check
                if not (0 <= current_x < self.WIDTH and 0 <= current_y < self.HEIGHT):
                    break
                
                # Check if reached water
                if self.map[current_y][current_x] in [self.SHALLOW_SEA, self.DEEP_SEA]:
                    river_path.append((current_x, current_y))
                    break
                
                river_path.append((current_x, current_y))
            
            # Place river, but convert land to beach where river meets land
            for i, (rx, ry) in enumerate(river_path):
                if self.map[ry][rx] in [self.PLAINS, self.FOREST]:
                    if i == 0 or i == len(river_path) - 1:
                        # Start or end of river
                        self.map[ry][rx] = self.RIVER
                    else:
                        self.map[ry][rx] = self.RIVER
    
    def _fix_beach_connectivity(self):
        """Ensure all beaches properly connect water to land."""
        new_map = [row[:] for row in self.map]
        
        # Add beaches where rivers meet land
        for y in range(self.HEIGHT):
            for x in range(self.WIDTH):
                if self.map[y][x] == self.RIVER:
                    # Add beaches adjacent to rivers
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.WIDTH and 0 <= ny < self.HEIGHT:
                                if self.map[ny][nx] in [self.PLAINS, self.FOREST]:
                                    new_map[ny][nx] = self.BEACH
        
        self.map = new_map
    
    def _validate_and_cleanup(self):
        """Validate and fix any rule violations."""
        new_map = [row[:] for row in self.map]
        
        for y in range(self.HEIGHT):
            for x in range(self.WIDTH):
                biome = self.map[y][x]
                
                # Rule: Sea and rivers may only touch beaches
                if biome in [self.DEEP_SEA, self.SHALLOW_SEA, self.RIVER]:
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.WIDTH and 0 <= ny < self.HEIGHT:
                                neighbor = self.map[ny][nx]
                                if neighbor in [self.PLAINS, self.FOREST, self.MOUNTAIN, self.PEAK]:
                                    # Add beach between water and land
                                    new_map[ny][nx] = self.BEACH
                
                # Rule: Beaches must touch water
                if biome == self.BEACH:
                    has_water = self._has_adjacent_water(x, y)
                    has_land = self._has_adjacent_biome(x, y, [self.PLAINS, self.FOREST, self.MOUNTAIN])
                    
                    if not has_water:
                        # Beach without water becomes plains
                        new_map[y][x] = self.PLAINS
                    elif not has_land and not self._has_adjacent_biome(x, y, [self.RIVER]):
                        # Beach without land becomes shallow water
                        new_map[y][x] = self.SHALLOW_SEA
                
                # Rule: Mountains may only touch plains or forests (and other mountains/peaks)
                if biome == self.MOUNTAIN:
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.WIDTH and 0 <= ny < self.HEIGHT:
                                neighbor = self.map[ny][nx]
                                if neighbor == self.BEACH:
                                    # Mountain touching beach - convert beach to plains
                                    new_map[ny][nx] = self.PLAINS
                                elif neighbor in [self.DEEP_SEA, self.SHALLOW_SEA, self.RIVER]:
                                    # Mountain touching water - add plains buffer
                                    new_map[y][x] = self.PLAINS
                
                # Rule: Peaks must be surrounded by mountains
                if biome == self.PEAK:
                    surrounded = True
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.WIDTH and 0 <= ny < self.HEIGHT:
                                if self.map[ny][nx] not in [self.MOUNTAIN, self.PEAK]:
                                    surrounded = False
                                    break
                        if not surrounded:
                            break
                    
                    if not surrounded:
                        new_map[y][x] = self.MOUNTAIN
        
        self.map = new_map
    
    # Helper methods
    def _has_adjacent_land(self, x, y):
        """Check if position has adjacent land."""
        return self._has_adjacent_biome(x, y, [self.PLAINS, self.FOREST, self.BEACH, self.MOUNTAIN, self.PEAK])
    
    def _has_adjacent_water(self, x, y):
        """Check if position has adjacent water."""
        return self._has_adjacent_biome(x, y, [self.DEEP_SEA, self.SHALLOW_SEA, self.RIVER])
    
    def _has_adjacent_biome(self, x, y, biomes):
        """Check if position has adjacent biome of given type."""
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.WIDTH and 0 <= ny < self.HEIGHT:
                    if self.map[ny][nx] in biomes:
                        return True
        return False
    
    def _has_beach_nearby(self, x, y, radius=3):
        """Check if there's a beach within given radius."""
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.WIDTH and 0 <= ny < self.HEIGHT:
                    if self.map[ny][nx] == self.BEACH:
                        return True
        return False
    
    def _find_nearest_water(self, x, y):
        """Find nearest water tile using breadth-first search."""
        max_search = 50
        for radius in range(1, max_search):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if abs(dx) == radius or abs(dy) == radius:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.WIDTH and 0 <= ny < self.HEIGHT:
                            if self.map[ny][nx] in [self.DEEP_SEA, self.SHALLOW_SEA]:
                                return (nx, ny)
        return None


class MapGeneratorGUI:
    """GUI for the map generator using tkinter."""
    
    def __init__(self, root):
        """Initialize the GUI."""
        self.root = root
        self.root.title("Video Game Map Generator")
        
        self.generator = MapGenerator()
        
        # Create canvas for map display
        canvas_width = MapGenerator.WIDTH * MapGenerator.PIXEL_SIZE
        canvas_height = MapGenerator.HEIGHT * MapGenerator.PIXEL_SIZE
        
        self.canvas = Canvas(
            root, 
            width=canvas_width, 
            height=canvas_height,
            bg='black'
        )
        self.canvas.pack(padx=10, pady=10)
        
        # Create regenerate button
        self.btn_regenerate = Button(
            root,
            text="Regenerate Map",
            command=self.regenerate_map,
            font=('Arial', 12, 'bold'),
            bg='#4CAF50',
            fg='white',
            padx=20,
            pady=10
        )
        self.btn_regenerate.pack(pady=10)
        
        # Generate initial map
        self.regenerate_map()
    
    def regenerate_map(self):
        """Regenerate and display a new map."""
        # Generate new map
        game_map = self.generator.generate_map()
        
        # Clear canvas
        self.canvas.delete('all')
        
        # Draw map
        pixel_size = MapGenerator.PIXEL_SIZE
        for y in range(MapGenerator.HEIGHT):
            for x in range(MapGenerator.WIDTH):
                biome = game_map[y][x]
                color = MapGenerator.COLORS[biome]
                
                x1 = x * pixel_size
                y1 = y * pixel_size
                x2 = x1 + pixel_size
                y2 = y1 + pixel_size
                
                self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=color,
                    outline=color
                )
        
        # Update display
        self.root.update()


def main():
    """Main function to run the map generator."""
    t1 = time.perf_counter()
    root = tk.Tk()
    app = MapGeneratorGUI(root)
    t2 = time.perf_counter()
    print(t2-t1)
    root.mainloop()


if __name__ == "__main__":
    main()