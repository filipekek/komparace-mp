import tkinter as tk
import random
import time
from math import floor

# --- Configuration ---
MAP_WIDTH = 520
MAP_HEIGHT = 520
SEED = random.randint(0, 10000)

# Colors
COLOR_DEEP_SEA = "#00008B"   # Dark Blue
COLOR_WATER    = "#ADD8E6"   # Light Blue (Rivers/Shallows)
COLOR_BEACH    = "#FFFF00"   # Yellow
COLOR_PLAINS   = "#90EE90"   # Light Green
COLOR_FOREST   = "#006400"   # Dark Green
COLOR_MOUNTAIN = "#808080"   # Gray
COLOR_SNOW     = "#FFFFFF"   # White

class NoiseGenerator:
    """
    A robust implementation of Fractal Value Noise (simplified Perlin-like behavior).
    Optimized for Python performance by pre-calculating grids and using
    bilinear interpolation without external dependencies.
    """
    def __init__(self, width, height, seed):
        self.width = width
        self.height = height
        self.seed = seed

    def generate_noise_grid(self, grid_size):
        """Creates a coarse grid of random values."""
        random.seed(self.seed + grid_size) # unique seed per octave
        return [[random.random() for _ in range(grid_size + 1)] for _ in range(grid_size + 1)]

    def get_value(self, x, y, grid, grid_size):
        """Bilinear interpolation to fetch smooth noise value."""
        # Map x,y (0..width) to grid coordinates
        gx = x / self.width * grid_size
        gy = y / self.height * grid_size
        
        x0 = int(gx)
        y0 = int(gy)
        x1 = min(x0 + 1, grid_size)
        y1 = min(y0 + 1, grid_size)
        
        sx = gx - x0
        sy = gy - y0
        
        # Fetch corner values
        v00 = grid[y0][x0]
        v10 = grid[y0][x1]
        v01 = grid[y1][x0]
        v11 = grid[y1][x1]
        
        # Interpolate
        top = v00 + (v10 - v00) * sx
        bottom = v01 + (v11 - v01) * sx
        return top + (bottom - top) * sy

    def generate_fractal_noise(self, octaves=4, persistence=0.5):
        """
        Generates a composite noise map by layering multiple octaves.
        Returns a flat list of float values 0.0-1.0.
        """
        # Pre-generate grids for all octaves to speed up the pixel loop
        grids = []
        frequencies = []
        amplitude = 1.0
        total_amplitude = 0.0
        
        curr_freq = 4 # Starting frequency
        
        for _ in range(octaves):
            grids.append(self.generate_noise_grid(curr_freq))
            frequencies.append(curr_freq)
            total_amplitude += amplitude
            amplitude *= persistence
            curr_freq *= 2

        # Generate pixel data
        map_data = []
        
        # Optimization: Local variable lookup is faster in Python loops
        w, h = self.width, self.height
        
        for y in range(h):
            row = []
            for x in range(w):
                noise_val = 0.0
                amp = 1.0
                
                for i in range(octaves):
                    # Inline the get_value logic partially for performance if needed,
                    # but function call is cleaner for readability.
                    val = self.get_value(x, y, grids[i], frequencies[i])
                    noise_val += val * amp
                    amp *= persistence
                
                row.append(noise_val / total_amplitude)
            map_data.append(row)
            
        return map_data

class MapApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Biome Map Generator")
        self.root.resizable(False, False)

        # UI Setup
        self.canvas = tk.Canvas(root, width=MAP_WIDTH, height=MAP_HEIGHT, bg="black")
        self.canvas.pack()

        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack(fill=tk.X, pady=5)
        
        self.btn_gen = tk.Button(self.btn_frame, text="Regenerate World", command=self.generate_map)
        self.btn_gen.pack()

        self.status_label = tk.Label(self.btn_frame, text="Ready")
        self.status_label.pack()

        # Generate initial map
        self.root.after(100, self.generate_map)

    def determine_color(self, height, moisture):
        """
        Applies the specific biome rules based on height and moisture.
        HEIGHT THRESHOLDS:
        0.00 - 0.35 : Deep Sea
        0.35 - 0.45 : Water (Shallow/River) -> Touches Deep Sea or Beach
        0.45 - 0.50 : Beach                 -> Touches Water or Land
        0.50 - 0.80 : Land (Plain/Forest)   -> Touches Beach or Mountain
        0.80 - 0.90 : Mountain              -> Touches Land or Snow
        0.90 - 1.00 : Snow                  -> Touches Mountain
        """
        
        if height < 0.35:
            return COLOR_DEEP_SEA
        elif height < 0.45:
            return COLOR_WATER
        elif height < 0.50:
            return COLOR_BEACH
        elif height < 0.80:
            # Land Biomes split by moisture
            if moisture > 0.55: # Wetter -> Forest
                return COLOR_FOREST
            else:               # Drier -> Plains
                return COLOR_PLAINS
        elif height < 0.92:
            return COLOR_MOUNTAIN
        else:
            return COLOR_SNOW

    def generate_map(self):
        self.status_label.config(text="Generating... please wait.")
        self.root.update()
        
        start_time = time.time()
        
        # New seed
        seed = random.randint(0, 100000)
        
        # 1. Generate Height Map (Terrain shape)
        # Using 6 octaves creates detailed coastlines and mountains
        h_gen = NoiseGenerator(MAP_WIDTH, MAP_HEIGHT, seed)
        height_map = h_gen.generate_fractal_noise(octaves=6, persistence=0.5)

        # 2. Generate Moisture Map (Forest vs Plains distribution)
        # Using fewer octaves (4) makes biomes large and consistent
        m_gen = NoiseGenerator(MAP_WIDTH, MAP_HEIGHT, seed + 500)
        moisture_map = m_gen.generate_fractal_noise(octaves=4, persistence=0.5)

        # 3. Create Image Data
        # Tkinter PhotoImage accepts a list of lists of hex colors,
        # but creating a string in PPM format is significantly faster for large images.
        
        # Header for PPM format (P6 is binary, P3 is ascii. Tkinter supports P3 via data string)
        # We will use the standard hex-list method optimized with a single "put" call 
        # because constructing raw PPM strings in pure python can be complex to debug.
        # To optimize "put", we construct the matrix first.
        
        pixels = []
        for y in range(MAP_HEIGHT):
            row = []
            for x in range(MAP_WIDTH):
                h = height_map[y][x]
                m = moisture_map[y][x]
                color = self.determine_color(h, m)
                row.append(color)
            # Tkinter expects row data to be a tuple/list structure string or similar
            # Ideally: row_str = "{" + " ".join(row) + "}"
            pixels.append("{" + " ".join(row) + "}")

        # Join all rows to match Tkinter's expected tcl format for mass upload
        img_data = " ".join(pixels)
        
        self.img = tk.PhotoImage(width=MAP_WIDTH, height=MAP_HEIGHT)
        self.img.put(img_data)
        
        self.canvas.create_image(0, 0, image=self.img, anchor=tk.NW)
        
        elapsed = time.time() - start_time
        self.status_label.config(text=f"Generation Complete ({elapsed:.2f}s)")

if __name__ == "__main__":
    root = tk.Tk()
    app = MapApp(root)
    root.mainloop()