import tkinter as tk
import random
import math
import time

# --- Configuration ---
MAP_WIDTH = 240
MAP_HEIGHT = 160
PIXEL_SCALE = 4  # Enlarge pixels for clearer display
SEED_OFFSET = 0  # Helps randomize generation

# --- Colors (Hex) ---
C_DEEP_SEA = "#1e3f66"    # Dark Blue
C_SHALLOW = "#4fa4b8"     # Light Blue
C_BEACH = "#f4e482"       # Yellow
C_PLAINS = "#98cf6f"      # Light Green
C_FOREST = "#2e6b36"      # Dark Green
C_MOUNTAIN = "#707070"    # Gray
C_SNOW = "#ffffff"        # White

class NoiseGenerator:
    """
    A pure Python implementation of Value Noise with Fractal Brownian Motion.
    Used to generate smooth height and moisture maps.
    """
    def __init__(self, width, height, scale=20.0, octaves=4):
        self.width = width
        self.height = height
        self.scale = scale
        self.octaves = octaves
        self.perm = [random.randint(0, 255) for _ in range(512)]  # Permutation table

    def _lerp(self, a, b, t):
        """Linear Interpolation"""
        return a + t * (b - a)

    def _fade(self, t):
        """Smoothstep fade function for smoother transitions"""
        return t * t * t * (t * (t * 6 - 15) + 10)

    def _grad(self, hash_val, x, y):
        """
        Gradient function. 
        For value noise, we can simply return a value derived from hashing the coordinates.
        """
        h = hash_val & 15
        grad = 1 + (h & 7)  # Gradient value between 1 and 8
        if (h & 8): grad = -grad
        return grad

    def noise(self, x, y):
        """Calculates value noise at coordinates x, y"""
        X = int(math.floor(x)) & 255
        Y = int(math.floor(y)) & 255
        
        # Relative x, y in the grid cell
        xf = x - math.floor(x)
        yf = y - math.floor(y)

        u = self._fade(xf)
        v = self._fade(yf)

        # Hash coordinates of the 4 square corners
        p = self.perm
        aa = p[p[X] + Y]
        ab = p[p[X] + Y + 1]
        ba = p[p[X + 1] + Y]
        bb = p[p[X + 1] + Y + 1]

        # Interpolate
        # Note: This is a simplified Value Noise implementation for performance
        # We use the hashed value directly as the random height at that grid point
        # normalized to 0.0 - 1.0 range approximate
        
        def val(h): return (h % 256) / 255.0
        
        res = self._lerp(
            self._lerp(val(aa), val(ba), u),
            self._lerp(val(ab), val(bb), u),
            v
        )
        return res

    def fbm(self, x, y):
        """Fractal Brownian Motion: Layering noise for detail"""
        value = 0.0
        amplitude = 0.5
        frequency = 1.0
        
        for _ in range(self.octaves):
            value += self.noise(x * frequency / self.scale, y * frequency / self.scale) * amplitude
            amplitude *= 0.5
            frequency *= 2.0
            
        return value

class MapApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Procedural Biome Map Generator")
        
        # Calculate window size based on scale
        self.win_w = MAP_WIDTH * PIXEL_SCALE
        self.win_h = MAP_HEIGHT * PIXEL_SCALE
        
        # UI Setup
        self.canvas = tk.Canvas(root, width=self.win_w, height=self.win_h, bg="black", highlightthickness=0)
        self.canvas.pack()
        
        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack(fill=tk.X, pady=5)
        
        self.btn_regen = tk.Button(self.btn_frame, text="Regenerate Map", command=self.regenerate, font=("Arial", 12, "bold"))
        self.btn_regen.pack()

        # Status Label
        self.lbl_status = tk.Label(self.btn_frame, text="Ready", fg="gray")
        self.lbl_status.pack()

        # Initial Generation
        self.regenerate()

    def determine_color(self, height, moisture):
        """
        Logic for Biome Rules:
        Deep Water < Shallow < Beach < Plains/Forest < Mountain < Snow
        """
        
        # 1. WATER & BEACH
        # Rules: Sea/Rivers -> Beach -> Land
        if height < 0.35:
            return C_DEEP_SEA
        if height < 0.42:
            return C_SHALLOW
        if height < 0.47:
            # Beach acts as the buffer between Water and Land
            return C_BEACH
        
        # 2. LAND (Plains/Forest)
        # Rules: Connects Beach to Mountain
        if height < 0.70:
            # Use moisture map to decide between Forest and Plains
            # Forest requires more moisture; Plains are drier
            if moisture > 0.55:
                return C_FOREST
            else:
                return C_PLAINS
                
        # 3. MOUNTAIN
        # Rules: Touches Plains/Forest, surrounds High Altitude
        if height < 0.88:
            return C_MOUNTAIN
            
        # 4. HIGH ALTITUDE
        # Rules: Surrounded by Mountain (implied by height logic > 0.88)
        return C_SNOW

    def regenerate(self):
        self.lbl_status.config(text="Generating...")
        self.root.update_idletasks() # Force UI update
        
        start_time = time.time()
        
        # Randomize seeds
        seed_h = random.randint(0, 10000)
        seed_m = random.randint(0, 10000)
        
        # Initialize Noise Generators
        # Height map: Larger scale for continents
        noise_h = NoiseGenerator(MAP_WIDTH, MAP_HEIGHT, scale=40.0, octaves=5)
        noise_h.perm = [random.randint(0, 255) for _ in range(512)] # Reshuffle
        
        # Moisture map: Slightly different scale for vegetation variation
        noise_m = NoiseGenerator(MAP_WIDTH, MAP_HEIGHT, scale=50.0, octaves=3)
        noise_m.perm = [random.randint(0, 255) for _ in range(512)] # Reshuffle

        # --- Generate Pixel Data (PPM Format) ---
        # PPM Format (P6) header: "P6 width height 255 "
        # We use a list of strings for hex colors to build the data block
        
        pixel_data = []
        
        # We need to normalize the noise results roughly to 0..1 for easier thresholds
        # Since FBM isn't perfectly 0..1, we can clamp or scale. 
        # For this implementation, the raw output is usually between 0.2 and 0.8 roughly.
        
        for y in range(MAP_HEIGHT):
            row_colors = []
            for x in range(MAP_WIDTH):
                # Get noise values (coordinate + seed offset)
                h_val = noise_h.fbm(x + seed_h, y + seed_h)
                m_val = noise_m.fbm(x + seed_m, y + seed_m)
                
                # Normalize/Adjust noise to fit 0.0 - 1.0 better
                # The simple Value Noise + FBM tends to cluster in the middle.
                # We expand it slightly to hit the extremes (Deep Sea / Snow)
                h_val = (h_val - 0.2) * 1.8 
                
                hex_color = self.determine_color(h_val, m_val)
                
                # Convert hex to RGB tuple for PPM
                r = int(hex_color[1:3], 16)
                g = int(hex_color[3:5], 16)
                b = int(hex_color[5:7], 16)
                
                # For PPM P6, we need bytes. But Tkinter PhotoImage also accepts
                # a simpler string format: "{#RRGGBB #RRGGBB ...} {...}"
                # This is slower for massive images but safest for pure Python string manip.
                # HOWEVER, the fastest pure python way is writing PPM P6 bytes.
                
                row_colors.append((r, g, b))
            pixel_data.append(row_colors)

        # Construct PPM Header and Body
        # Header: P6 <width> <height> 255\n
        header = f"P6 {MAP_WIDTH} {MAP_HEIGHT} 255\n".encode('ascii')
        
        # Body: binary RGB data
        body_bytes = bytearray()
        for row in pixel_data:
            for r, g, b in row:
                body_bytes.append(max(0, min(255, r)))
                body_bytes.append(max(0, min(255, g)))
                body_bytes.append(max(0, min(255, b)))
                
        ppm_data = header + body_bytes

        # Create Image
        self.img = tk.PhotoImage(width=MAP_WIDTH, height=MAP_HEIGHT, data=ppm_data, format='PPM')
        
        # Scale Image (Zoom)
        # Tkinter's zoom repeats pixels, creating the "retro/pixelated" look requested
        self.scaled_img = self.img.zoom(PIXEL_SCALE, PIXEL_SCALE)
        
        # Update Canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.scaled_img, anchor=tk.NW)
        
        elapsed = time.time() - start_time
        self.lbl_status.config(text=f"Generated in {elapsed:.2f}s | Resolution: {MAP_WIDTH}x{MAP_HEIGHT}")

if __name__ == "__main__":
    t1 = time.perf_counter()
    root = tk.Tk()
    # Prevent window resizing to keep layout tight
    root.resizable(False, False)
    app = MapApp(root)
    t2 = time.perf_counter()
    print(t2-t1)
    root.mainloop()