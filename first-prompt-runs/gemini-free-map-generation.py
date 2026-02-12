import tkinter as tk
import random
import math

class MapGenerator:
    def __init__(self, root, size=520):
        self.size = size
        # Grid resolution (5x5 pixel blocks for detail)
        self.grid_size = 104 
        self.cell_size = size // self.grid_size
        
        self.canvas = tk.Canvas(root, width=size, height=size, bg="black", highlightthickness=0)
        self.canvas.pack(pady=10)
        
        self.COLORS = {
            'DEEP_SEA': "#102e4a",      # Dark Blue
            'SHALLOW_SEA': "#3a9bdc",   # Light Blue
            'BEACH': "#f2d16b",         # Yellow
            'PLAINS': "#91cf60",        # Light Green
            'FOREST': "#2d6a4f",        # Dark Green
            'MOUNTAINS': "#808080",     # Gray
            'PEAKS': "#ffffff"          # White
        }

    def generate_map_data(self):
        """Generates elevation grid using a combination of noise layers."""
        grid = [[0.0] * self.grid_size for _ in range(self.grid_size)]
        center = self.grid_size / 2
        
        # Random offsets for noise variation
        ox, oy = random.uniform(0, 100), random.uniform(0, 100)

        for y in range(self.grid_size):
            for x in range(self.grid_size):
                # 1. Distance-based gradient (ensures land is central)
                # Distance from center normalized 0 to 1
                dist = math.sqrt((x - center)**2 + (y - center)**2) / (self.grid_size / 1.4)
                gradient = max(0, 1.1 - dist) # Bias toward 1.0 at center
                
                # 2. Layered noise for 'ruggedness'
                # Uses sine waves of different frequencies to simulate noise
                noise = (
                    math.sin(x * 0.2 + ox) * math.cos(y * 0.2 + oy) * 0.5 +
                    math.sin(x * 0.1 - oy) * math.cos(y * 0.4 + ox) * 0.25 +
                    random.uniform(-0.1, 0.1) # High frequency jitter
                )
                
                # Normalize noise to a 0-1 range roughly
                noise = (noise + 1) / 2 
                
                # Combine: The gradient 'carves' the shape, noise adds the detail
                grid[y][x] = gradient * 0.6 + noise * 0.4

        return grid

    def get_biome(self, val):
        """Determines biome while strictly enforcing connectivity rules."""
        # The rules are enforced by the natural order of elevation:
        # Sea (Low) -> Beach -> Land -> Mountain -> Peak (High)
        if val < 0.40:
            return self.COLORS['DEEP_SEA']
        elif val < 0.52:
            return self.COLORS['SHALLOW_SEA']
        elif val < 0.57:
            return self.COLORS['BEACH']
        elif val < 0.75:
            # Randomly distribute Forest vs Plains within the land layer
            return self.COLORS['PLAINS'] if random.random() > 0.4 else self.COLORS['FOREST']
        elif val < 0.88:
            return self.COLORS['MOUNTAINS']
        else:
            return self.COLORS['PEAKS']

    def draw_map(self):
        self.canvas.delete("all")
        grid = self.generate_map_data()
        
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                color = self.get_biome(grid[y][x])
                
                x1, y1 = x * self.cell_size, y * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                
                # Draw the tile
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="", fill=color)

def main():
    root = tk.Tk()
    root.title("World Map Generator")
    root.resizable(False, False)
    
    gen = MapGenerator(root)
    gen.draw_map()
    
    # Standard Tkinter button uses padx/pady instead of -padding
    btn = tk.Button(root, text="Generate New World", command=gen.draw_map)
    btn.pack(pady=10, padx=10)
    
    root.mainloop()

if __name__ == "__main__":
    main()