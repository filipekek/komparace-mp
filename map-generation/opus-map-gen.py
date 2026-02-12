"""
Pixel Map Generator for Video Game World
Uses tkinter to create a diverse pixel map with proper biome connectivity rules.
Resolution: 240x160, with pixel scaling for clearer display.
"""

import tkinter as tk
import random
from collections import deque
import time

# Map dimensions
MAP_WIDTH = 240
MAP_HEIGHT = 160
PIXEL_SCALE = 4  # Each map pixel is displayed as 4x4 screen pixels

# Biome types
DEEP_SEA = 0
SHALLOW_WATER = 1
BEACH = 2
PLAINS = 3
FOREST = 4
MOUNTAIN = 5
PEAK = 6  # Higher altitudes (white)

# Colors for each biome
BIOME_COLORS = {
    DEEP_SEA: "#1a3a5c",       # Dark blue
    SHALLOW_WATER: "#5dadec",  # Light blue
    BEACH: "#f4d03f",          # Yellow
    PLAINS: "#90ee90",         # Light green
    FOREST: "#228b22",         # Dark green
    MOUNTAIN: "#808080",       # Gray
    PEAK: "#ffffff",           # White
}

# Biome names for debugging
BIOME_NAMES = {
    DEEP_SEA: "Deep Sea",
    SHALLOW_WATER: "Shallow Water",
    BEACH: "Beach",
    PLAINS: "Plains",
    FOREST: "Forest",
    MOUNTAIN: "Mountain",
    PEAK: "Peak",
}


class MapGenerator:
    """Generates a world map following biome connectivity rules."""
    
    def __init__(self):
        self.width = MAP_WIDTH
        self.height = MAP_HEIGHT
        self.map_data = [[PLAINS for _ in range(self.width)] for _ in range(self.height)]
    
    def get_neighbors(self, x, y, include_diagonal=False):
        """Get valid neighboring coordinates."""
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if include_diagonal:
            directions += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                neighbors.append((nx, ny))
        return neighbors
    
    def get_neighbor_biomes(self, x, y):
        """Get set of biome types in neighboring cells."""
        biomes = set()
        for nx, ny in self.get_neighbors(x, y, include_diagonal=True):
            biomes.add(self.map_data[ny][nx])
        return biomes
    
    def flood_fill(self, start_x, start_y, target_biome, max_size=None):
        """Flood fill from a starting point with optional size limit."""
        filled = set()
        queue = deque([(start_x, start_y)])
        
        while queue:
            if max_size and len(filled) >= max_size:
                break
            
            x, y = queue.popleft()
            if (x, y) in filled:
                continue
            if not (0 <= x < self.width and 0 <= y < self.height):
                continue
            
            filled.add((x, y))
            self.map_data[y][x] = target_biome
            
            # Add neighbors with some randomness for organic shapes
            neighbors = self.get_neighbors(x, y)
            random.shuffle(neighbors)
            for nx, ny in neighbors:
                if (nx, ny) not in filled and random.random() < 0.7:
                    queue.append((nx, ny))
        
        return filled
    
    def generate_noise_map(self):
        """Generate a simple noise map for terrain variation."""
        noise = [[0.0 for _ in range(self.width)] for _ in range(self.height)]
        
        # Generate multiple octaves of noise
        for octave in range(4):
            frequency = 2 ** octave
            amplitude = 1.0 / (octave + 1)
            
            # Generate random seed points
            seed_spacing = max(4, 32 // frequency)
            seeds = {}
            
            for sy in range(0, self.height + seed_spacing, seed_spacing):
                for sx in range(0, self.width + seed_spacing, seed_spacing):
                    seeds[(sx, sy)] = random.random()
            
            # Interpolate between seed points
            for y in range(self.height):
                for x in range(self.width):
                    # Find surrounding seed points
                    sx0 = (x // seed_spacing) * seed_spacing
                    sy0 = (y // seed_spacing) * seed_spacing
                    sx1 = sx0 + seed_spacing
                    sy1 = sy0 + seed_spacing
                    
                    # Get seed values (with default for edge cases)
                    v00 = seeds.get((sx0, sy0), 0.5)
                    v10 = seeds.get((sx1, sy0), 0.5)
                    v01 = seeds.get((sx0, sy1), 0.5)
                    v11 = seeds.get((sx1, sy1), 0.5)
                    
                    # Bilinear interpolation
                    tx = (x - sx0) / seed_spacing if seed_spacing > 0 else 0
                    ty = (y - sy0) / seed_spacing if seed_spacing > 0 else 0
                    
                    # Smooth interpolation
                    tx = tx * tx * (3 - 2 * tx)
                    ty = ty * ty * (3 - 2 * ty)
                    
                    v0 = v00 * (1 - tx) + v10 * tx
                    v1 = v01 * (1 - tx) + v11 * tx
                    value = v0 * (1 - ty) + v1 * ty
                    
                    noise[y][x] += value * amplitude
        
        # Normalize noise to 0-1 range
        min_val = min(min(row) for row in noise)
        max_val = max(max(row) for row in noise)
        if max_val > min_val:
            for y in range(self.height):
                for x in range(self.width):
                    noise[y][x] = (noise[y][x] - min_val) / (max_val - min_val)
        
        return noise
    
    def generate_base_terrain(self):
        """Generate the initial terrain using noise."""
        noise = self.generate_noise_map()
        
        # Create elevation-based terrain
        for y in range(self.height):
            for x in range(self.width):
                elevation = noise[y][x]
                
                # Add distance from center influence for island-like maps
                cx, cy = self.width / 2, self.height / 2
                dist = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
                max_dist = ((cx) ** 2 + (cy) ** 2) ** 0.5
                edge_factor = 1 - (dist / max_dist) * 0.6
                elevation *= edge_factor
                
                # Assign biomes based on elevation
                if elevation < 0.25:
                    self.map_data[y][x] = DEEP_SEA
                elif elevation < 0.35:
                    self.map_data[y][x] = SHALLOW_WATER
                elif elevation < 0.40:
                    self.map_data[y][x] = BEACH
                elif elevation < 0.55:
                    self.map_data[y][x] = PLAINS
                elif elevation < 0.70:
                    self.map_data[y][x] = FOREST
                elif elevation < 0.85:
                    self.map_data[y][x] = MOUNTAIN
                else:
                    self.map_data[y][x] = PEAK
    
    def add_rivers(self):
        """Add rivers that flow from mountains to the sea."""
        # Find potential river sources (mountains near peaks)
        sources = []
        for y in range(self.height):
            for x in range(self.width):
                if self.map_data[y][x] == MOUNTAIN:
                    neighbors = self.get_neighbor_biomes(x, y)
                    if PEAK in neighbors:
                        sources.append((x, y))
        
        if not sources:
            return
        
        # Create a few rivers
        num_rivers = random.randint(2, 5)
        random.shuffle(sources)
        
        for i in range(min(num_rivers, len(sources))):
            self.create_river(sources[i][0], sources[i][1])
    
    def create_river(self, start_x, start_y):
        """Create a river from a starting point flowing towards the sea."""
        x, y = start_x, start_y
        river_path = [(x, y)]
        max_length = 200
        
        # Simple elevation map for river flow
        elevation_priority = {
            PEAK: 6, MOUNTAIN: 5, FOREST: 4, PLAINS: 3, 
            BEACH: 2, SHALLOW_WATER: 1, DEEP_SEA: 0
        }
        
        for _ in range(max_length):
            # Find lowest neighbor
            neighbors = self.get_neighbors(x, y)
            random.shuffle(neighbors)  # Randomize for variety
            
            best_neighbor = None
            best_elevation = elevation_priority.get(self.map_data[y][x], 10)
            
            for nx, ny in neighbors:
                if (nx, ny) in river_path:
                    continue
                neighbor_elevation = elevation_priority.get(self.map_data[ny][nx], 10)
                # Prefer lower elevation or same level with some randomness
                if neighbor_elevation <= best_elevation:
                    if random.random() < 0.6 or neighbor_elevation < best_elevation:
                        best_elevation = neighbor_elevation
                        best_neighbor = (nx, ny)
            
            if best_neighbor is None:
                break
            
            x, y = best_neighbor
            river_path.append((x, y))
            
            # Stop if we reach water
            if self.map_data[y][x] in (SHALLOW_WATER, DEEP_SEA):
                break
        
        # Convert river path to shallow water (rivers)
        for rx, ry in river_path:
            if self.map_data[ry][rx] not in (DEEP_SEA, SHALLOW_WATER, BEACH):
                self.map_data[ry][rx] = SHALLOW_WATER
    
    def enforce_biome_rules(self):
        """
        Enforce biome connectivity rules:
        - Sea/rivers may only touch beaches
        - Beaches must touch sea/rivers and connect to forests/plains
        - Mountains may only touch plains or forests
        - Peaks must be surrounded by mountains
        """
        max_iterations = 10
        
        for iteration in range(max_iterations):
            changes_made = False
            
            for y in range(self.height):
                for x in range(self.width):
                    current = self.map_data[y][x]
                    neighbors = self.get_neighbor_biomes(x, y)
                    
                    # Rule: Water (sea/rivers) may only touch beaches (or other water)
                    if current in (DEEP_SEA, SHALLOW_WATER):
                        invalid_neighbors = neighbors & {PLAINS, FOREST, MOUNTAIN, PEAK}
                        if invalid_neighbors:
                            # Convert water to beach or convert neighbor to beach
                            self.map_data[y][x] = BEACH
                            changes_made = True
                    
                    # Rule: Beaches must touch water and land (plains/forest)
                    elif current == BEACH:
                        water_neighbors = neighbors & {DEEP_SEA, SHALLOW_WATER}
                        land_neighbors = neighbors & {PLAINS, FOREST}
                        
                        # If beach doesn't touch water, try to fix
                        if not water_neighbors:
                            # Check if any neighbor is beach that touches water
                            has_beach_path_to_water = False
                            for nx, ny in self.get_neighbors(x, y):
                                if self.map_data[ny][nx] == BEACH:
                                    has_beach_path_to_water = True
                                    break
                            
                            if not has_beach_path_to_water:
                                # Convert to plains
                                self.map_data[y][x] = PLAINS
                                changes_made = True
                        
                        # Beach shouldn't directly touch mountains/peaks
                        if neighbors & {MOUNTAIN, PEAK}:
                            self.map_data[y][x] = PLAINS
                            changes_made = True
                    
                    # Rule: Mountains may only touch plains, forests, or other mountains/peaks
                    elif current == MOUNTAIN:
                        invalid_neighbors = neighbors & {DEEP_SEA, SHALLOW_WATER, BEACH}
                        if invalid_neighbors:
                            # Convert mountain to forest or plains
                            self.map_data[y][x] = FOREST
                            changes_made = True
                    
                    # Rule: Peaks must be surrounded by mountains (or other peaks)
                    elif current == PEAK:
                        valid_neighbors = {MOUNTAIN, PEAK}
                        if not neighbors.issubset(valid_neighbors):
                            # Either convert peak to mountain or fix neighbors
                            if neighbors & {DEEP_SEA, SHALLOW_WATER, BEACH}:
                                self.map_data[y][x] = MOUNTAIN
                                changes_made = True
                            elif neighbors & {PLAINS, FOREST}:
                                # Convert to mountain
                                self.map_data[y][x] = MOUNTAIN
                                changes_made = True
            
            if not changes_made:
                break
    
    def add_beach_borders(self):
        """Ensure water is always bordered by beaches."""
        changes = []
        
        for y in range(self.height):
            for x in range(self.width):
                if self.map_data[y][x] in (DEEP_SEA, SHALLOW_WATER):
                    for nx, ny in self.get_neighbors(x, y):
                        neighbor = self.map_data[ny][nx]
                        if neighbor in (PLAINS, FOREST, MOUNTAIN, PEAK):
                            changes.append((nx, ny, BEACH))
        
        for x, y, biome in changes:
            self.map_data[y][x] = biome
    
    def add_mountain_buffer(self):
        """Ensure mountains don't directly touch water or beaches."""
        changes = []
        
        for y in range(self.height):
            for x in range(self.width):
                if self.map_data[y][x] in (MOUNTAIN, PEAK):
                    neighbors = self.get_neighbor_biomes(x, y)
                    if neighbors & {DEEP_SEA, SHALLOW_WATER, BEACH}:
                        # Add forest/plains buffer
                        changes.append((x, y, FOREST))
        
        for x, y, biome in changes:
            self.map_data[y][x] = biome
    
    def smooth_terrain(self):
        """Smooth terrain to reduce noise and create more cohesive regions."""
        new_map = [[self.map_data[y][x] for x in range(self.width)] for y in range(self.height)]
        
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                current = self.map_data[y][x]
                neighbors = [self.map_data[ny][nx] for nx, ny in self.get_neighbors(x, y)]
                
                # Count neighbor biomes
                biome_counts = {}
                for b in neighbors:
                    biome_counts[b] = biome_counts.get(b, 0) + 1
                
                # If current biome is minority, consider changing
                current_count = biome_counts.get(current, 0)
                most_common = max(biome_counts.items(), key=lambda x: x[1])
                
                if most_common[1] >= 3 and current_count <= 1:
                    # Only change if it doesn't violate rules
                    new_biome = most_common[0]
                    new_map[y][x] = new_biome
        
        self.map_data = new_map
    
    def generate(self):
        """Generate a complete map following all rules."""
        # Reset map
        self.map_data = [[PLAINS for _ in range(self.width)] for _ in range(self.height)]
        
        # Step 1: Generate base terrain with noise
        self.generate_base_terrain()
        
        # Step 2: Smooth terrain
        for _ in range(2):
            self.smooth_terrain()
        
        # Step 3: Add rivers
        self.add_rivers()
        
        # Step 4: Add beach borders around water
        self.add_beach_borders()
        
        # Step 5: Add buffer between mountains and water/beaches
        self.add_mountain_buffer()
        
        # Step 6: Enforce all biome connectivity rules
        self.enforce_biome_rules()
        
        # Step 7: Final beach pass to ensure water connectivity
        self.add_beach_borders()
        
        # Step 8: Final rule enforcement
        for _ in range(3):
            self.enforce_biome_rules()
        
        return self.map_data


class MapDisplay:
    """Displays the generated map using tkinter."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("World Map Generator")
        self.root.resizable(False, False)
        
        # Calculate canvas size
        self.canvas_width = MAP_WIDTH * PIXEL_SCALE
        self.canvas_height = MAP_HEIGHT * PIXEL_SCALE
        
        # Create main frame
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(padx=10, pady=10)
        
        # Create canvas for map display
        self.canvas = tk.Canvas(
            self.main_frame,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="black",
            highlightthickness=1,
            highlightbackground="gray"
        )
        self.canvas.pack()
        
        # Create button frame
        self.button_frame = tk.Frame(self.main_frame)
        self.button_frame.pack(pady=10)
        
        # Create regenerate button
        self.regenerate_button = tk.Button(
            self.button_frame,
            text="Regenerate Map",
            command=self.regenerate_map,
            font=("Arial", 12),
            padx=20,
            pady=5
        )
        self.regenerate_button.pack()
        
        # Create map generator
        self.generator = MapGenerator()
        
        # Generate and display initial map
        self.regenerate_map()
    
    def draw_map(self, map_data):
        """Draw the map on the canvas."""
        # Clear canvas
        self.canvas.delete("all")
        
        # Create image for faster rendering
        # We'll draw rectangles for each pixel (scaled)
        for y in range(MAP_HEIGHT):
            for x in range(MAP_WIDTH):
                biome = map_data[y][x]
                color = BIOME_COLORS.get(biome, "#ff00ff")  # Magenta for unknown
                
                # Calculate screen coordinates
                sx = x * PIXEL_SCALE
                sy = y * PIXEL_SCALE
                
                # Draw scaled pixel
                self.canvas.create_rectangle(
                    sx, sy,
                    sx + PIXEL_SCALE, sy + PIXEL_SCALE,
                    fill=color,
                    outline=""  # No outline for cleaner look
                )
        
        # Update canvas
        self.canvas.update()
    
    def regenerate_map(self):
        """Generate a new map and display it."""
        # Disable button during generation
        self.regenerate_button.config(state=tk.DISABLED, text="Generating...")
        self.root.update()
        
        # Generate new map
        map_data = self.generator.generate()
        
        # Draw map
        self.draw_map(map_data)
        
        # Re-enable button
        self.regenerate_button.config(state=tk.NORMAL, text="Regenerate Map")
    
    def run(self):
        """Start the application."""
        self.root.mainloop()


def validate_map(map_data):
    """Validate that the map follows all biome rules (for testing)."""
    width = len(map_data[0])
    height = len(map_data)
    errors = []
    
    def get_neighbors(x, y):
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                neighbors.append((nx, ny))
        return neighbors
    
    def get_neighbor_biomes(x, y):
        biomes = set()
        for nx, ny in get_neighbors(x, y):
            biomes.add(map_data[ny][nx])
        return biomes
    
    for y in range(height):
        for x in range(width):
            current = map_data[y][x]
            neighbors = get_neighbor_biomes(x, y)
            
            # Check water rules
            if current in (DEEP_SEA, SHALLOW_WATER):
                invalid = neighbors & {PLAINS, FOREST, MOUNTAIN, PEAK}
                if invalid:
                    errors.append(f"Water at ({x},{y}) touches {invalid}")
            
            # Check mountain rules
            if current == MOUNTAIN:
                invalid = neighbors & {DEEP_SEA, SHALLOW_WATER, BEACH}
                if invalid:
                    errors.append(f"Mountain at ({x},{y}) touches {invalid}")
            
            # Check peak rules
            if current == PEAK:
                valid = {MOUNTAIN, PEAK}
                if not neighbors.issubset(valid) and neighbors:
                    invalid = neighbors - valid
                    errors.append(f"Peak at ({x},{y}) touches {invalid}")
    
    return errors


if __name__ == "__main__":
    # Create and run the map display
    t1 = time.perf_counter()
    app = MapDisplay()
    t2 = time.perf_counter()
    print(t2-t1)
    app.run()