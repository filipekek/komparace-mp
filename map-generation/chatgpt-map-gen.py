import random
from collections import deque
import tkinter as tk
import time

class WorldMapGenerator:
    # Base resolution (pixels) and display scale
    W, H = 240, 160
    SCALE = 4

    # Colors (hex)
    COLORS = {
        "SEA_DEEP": "#0b2a80",     # dark blue
        "SEA_SHALLOW": "#4aa3ff",  # light blue
        "RIVER": "#4aa3ff",        # light blue (same as shallow water)
        "BEACH": "#f2e56b",        # yellow
        "PLAIN": "#7bdc77",        # light green
        "FOREST": "#1f7a2e",       # dark green
        "MOUNTAIN": "#7f7f7f",     # gray
        "SNOW": "#ffffff",         # white
    }

    # Terrain codes (ints) for speed and clarity
    SEA_DEEP = 0
    SEA_SHALLOW = 1
    RIVER = 2
    BEACH = 3
    PLAIN = 4
    FOREST = 5
    MOUNTAIN = 6
    SNOW = 7

    CODE_TO_COLOR = {
        SEA_DEEP: COLORS["SEA_DEEP"],
        SEA_SHALLOW: COLORS["SEA_SHALLOW"],
        RIVER: COLORS["RIVER"],
        BEACH: COLORS["BEACH"],
        PLAIN: COLORS["PLAIN"],
        FOREST: COLORS["FOREST"],
        MOUNTAIN: COLORS["MOUNTAIN"],
        SNOW: COLORS["SNOW"],
    }

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    @staticmethod
    def _in_bounds(x, y, w, h):
        return 0 <= x < w and 0 <= y < h

    @staticmethod
    def _neighbors4(x, y):
        return ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))

    @staticmethod
    def _neighbors8(x, y):
        return (
            (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1),
            (x - 1, y - 1), (x + 1, y - 1), (x - 1, y + 1), (x + 1, y + 1)
        )

    def _smooth(self, grid, passes=4):
        """Simple box blur smoothing (pure Python)."""
        w, h = self.W, self.H
        for _ in range(passes):
            new = [[0.0] * w for _ in range(h)]
            for y in range(h):
                y0 = max(0, y - 1)
                y1 = min(h - 1, y + 1)
                for x in range(w):
                    x0 = max(0, x - 1)
                    x1 = min(w - 1, x + 1)
                    s = 0.0
                    c = 0
                    for yy in range(y0, y1 + 1):
                        row = grid[yy]
                        for xx in range(x0, x1 + 1):
                            s += row[xx]
                            c += 1
                    new[y][x] = s / c
            grid = new
        return grid

    def _heightmap(self):
        """Create a 0..1 heightmap with some large-scale variation."""
        w, h = self.W, self.H
        base = [[self.rng.random() for _ in range(w)] for _ in range(h)]
        base = self._smooth(base, passes=5)

        # Add a second layer for diversity
        detail = [[self.rng.random() for _ in range(w)] for _ in range(h)]
        detail = self._smooth(detail, passes=2)

        # Combine with weights
        hm = [[0.0] * w for _ in range(h)]
        for y in range(h):
            for x in range(w):
                v = 0.78 * base[y][x] + 0.22 * detail[y][x]
                hm[y][x] = v

        # Normalize to 0..1
        mn = min(min(r) for r in hm)
        mx = max(max(r) for r in hm)
        span = (mx - mn) if mx > mn else 1.0
        for y in range(h):
            for x in range(w):
                hm[y][x] = (hm[y][x] - mn) / span
        return hm

    def _distance_field(self, sources, blocked=None):
        """
        Multi-source BFS for Manhattan distance.
        sources: iterable of (x,y)
        blocked: set of (x,y) not traversable (optional)
        """
        w, h = self.W, self.H
        dist = [[10**9] * w for _ in range(h)]
        q = deque()

        for (x, y) in sources:
            if self._in_bounds(x, y, w, h):
                dist[y][x] = 0
                q.append((x, y))

        blocked = blocked or set()

        while q:
            x, y = q.popleft()
            nd = dist[y][x] + 1
            for nx, ny in self._neighbors4(x, y):
                if not self._in_bounds(nx, ny, w, h):
                    continue
                if (nx, ny) in blocked:
                    continue
                if nd < dist[ny][nx]:
                    dist[ny][nx] = nd
                    q.append((nx, ny))
        return dist

    def _carve_river(self, terrain, height, start, sea_codes, forbidden_codes, max_len=4000):
        """
        Carve a river from start toward the sea by greedy descent with mild randomness.
        Returns True if a river reached sea (adjacent to sea), otherwise False.
        """
        w, h = self.W, self.H
        x, y = start
        visited = set()

        def is_sea(x0, y0):
            return terrain[y0][x0] in sea_codes

        for _ in range(max_len):
            if (x, y) in visited:
                return False
            visited.add((x, y))

            # If we are adjacent to sea, stop and keep the current cell as river
            adjacent_to_sea = False
            for nx, ny in self._neighbors4(x, y):
                if self._in_bounds(nx, ny, w, h) and is_sea(nx, ny):
                    adjacent_to_sea = True
                    break

            # Mark current cell as river (but do not overwrite sea)
            if terrain[y][x] not in sea_codes:
                terrain[y][x] = self.RIVER

            if adjacent_to_sea:
                return True

            # Choose next step among neighbors: prefer lower height, avoid forbidden
            candidates = []
            for nx, ny in self._neighbors4(x, y):
                if not self._in_bounds(nx, ny, w, h):
                    continue
                tc = terrain[ny][nx]
                if tc in forbidden_codes:
                    continue
                # Allow stepping into beach/plain/forest/river (not mountains/snow/sea)
                # Add tiny random noise to avoid straight lines
                score = height[ny][nx] + self.rng.random() * 0.01
                candidates.append((score, nx, ny))

            if not candidates:
                return False

            candidates.sort(key=lambda t: t[0])
            # Mostly take best, sometimes the second best for variation
            if len(candidates) >= 2 and self.rng.random() < 0.15:
                _, x, y = candidates[1]
            else:
                _, x, y = candidates[0]

        return False

    def _apply_beaches(self, terrain):
        """All land adjacent to any water becomes beach, except mountains/snow (which should not be adjacent)."""
        w, h = self.W, self.H
        water = {self.SEA_DEEP, self.SEA_SHALLOW, self.RIVER}
        for y in range(h):
            for x in range(w):
                if terrain[y][x] in water:
                    continue
                if terrain[y][x] in (self.MOUNTAIN, self.SNOW):
                    continue
                for nx, ny in self._neighbors4(x, y):
                    if self._in_bounds(nx, ny, w, h) and terrain[ny][nx] in water:
                        terrain[y][x] = self.BEACH
                        break

    def _validate(self, terrain):
        """
        Validate the constraints:
        - Water (sea/river) touches only beaches (on land-water boundaries).
        - Beaches touch water and connect inland to plains/forests.
        - Mountains touch only plains/forests (no water, no beach, no snow adjacency constraints beyond snow check).
        - Snow cells are surrounded by mountain ranges (8-neighborhood).
        """
        w, h = self.W, self.H
        water = {self.SEA_DEEP, self.SEA_SHALLOW, self.RIVER}

        for y in range(h):
            for x in range(w):
                t = terrain[y][x]

                if t in water:
                    # Adjacent land must be BEACH (or water)
                    for nx, ny in self._neighbors4(x, y):
                        if not self._in_bounds(nx, ny, w, h):
                            continue
                        nt = terrain[ny][nx]
                        if nt in water:
                            continue
                        if nt != self.BEACH:
                            return False

                elif t == self.BEACH:
                    # Must touch water
                    touches_water = False
                    touches_inland = False
                    for nx, ny in self._neighbors4(x, y):
                        if not self._in_bounds(nx, ny, w, h):
                            continue
                        nt = terrain[ny][nx]
                        if nt in water:
                            touches_water = True
                        if nt in (self.PLAIN, self.FOREST):
                            touches_inland = True
                        # Mountains must not touch beaches (enforced via mountain rule too)
                        if nt in (self.MOUNTAIN, self.SNOW):
                            return False
                    if not touches_water:
                        return False
                    if not touches_inland:
                        return False

                elif t == self.MOUNTAIN:
                    for nx, ny in self._neighbors4(x, y):
                        if not self._in_bounds(nx, ny, w, h):
                            continue
                        nt = terrain[ny][nx]
                        if nt in (self.MOUNTAIN, self.SNOW):
                            continue
                        if nt not in (self.PLAIN, self.FOREST):
                            return False

                elif t == self.SNOW:
                    # Must be surrounded by mountains (8-neighborhood)
                    for nx, ny in self._neighbors8(x, y):
                        if not self._in_bounds(nx, ny, w, h):
                            return False  # edge snow is not allowed (can't be fully surrounded)
                        if terrain[ny][nx] != self.MOUNTAIN:
                            return False

        return True

    def generate(self, max_attempts=30):
        """
        Generate a valid map. Retries a limited number of times if constraints fail.
        Returns: terrain grid (H x W) of codes.
        """
        for _ in range(max_attempts):
            height = self._heightmap()
            w, h = self.W, self.H

            # Initialize terrain as land; later assign sea
            terrain = [[self.PLAIN] * w for _ in range(h)]

            # Sea level controls land/sea ratio (tune slightly)
            sea_level = 0.44 + self.rng.random() * 0.05

            # Assign sea by height threshold
            for y in range(h):
                for x in range(w):
                    if height[y][x] < sea_level:
                        terrain[y][x] = self.SEA_DEEP  # temp; shallow will be refined

            # Refine shallow vs deep: shallow = any sea cell adjacent to land
            for y in range(h):
                for x in range(w):
                    if terrain[y][x] == self.SEA_DEEP:
                        for nx, ny in self._neighbors4(x, y):
                            if self._in_bounds(nx, ny, w, h) and terrain[ny][nx] != self.SEA_DEEP:
                                # neighbor is land (or later river), so this is shallow sea
                                terrain[y][x] = self.SEA_SHALLOW
                                break

            # Distance to sea (water) BEFORE rivers
            sea_cells = [(x, y) for y in range(h) for x in range(w) if terrain[y][x] in (self.SEA_DEEP, self.SEA_SHALLOW)]
            dist_to_sea = self._distance_field(sea_cells)

            # Mountains: only far enough from sea so they won't touch beaches
            mountain_level = 0.74 + self.rng.random() * 0.03
            snow_level = 0.86 + self.rng.random() * 0.02

            for y in range(h):
                for x in range(w):
                    if terrain[y][x] in (self.SEA_DEEP, self.SEA_SHALLOW):
                        continue
                    # Keep a buffer so mountains do not end up adjacent to beaches
                    if dist_to_sea[y][x] >= 2 and height[y][x] >= mountain_level:
                        terrain[y][x] = self.MOUNTAIN

            # Snow: only if completely surrounded by mountains (8 neighbors)
            # First mark candidates by height, then enforce the surround rule.
            snow_candidates = []
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    if terrain[y][x] == self.MOUNTAIN and height[y][x] >= snow_level:
                        snow_candidates.append((x, y))
            for x, y in snow_candidates:
                if all(terrain[ny][nx] == self.MOUNTAIN for nx, ny in self._neighbors8(x, y)):
                    terrain[y][x] = self.SNOW

            # Remaining land: forest vs plain using another smoothed noise
            moisture = [[self.rng.random() for _ in range(w)] for _ in range(h)]
            moisture = self._smooth(moisture, passes=3)
            for y in range(h):
                for x in range(w):
                    if terrain[y][x] in (self.SEA_DEEP, self.SEA_SHALLOW, self.MOUNTAIN, self.SNOW):
                        continue
                    # Forest probability increases slightly with moisture and with moderate elevation
                    m = moisture[y][x]
                    elev = height[y][x]
                    p_forest = 0.35 + 0.35 * m + 0.10 * (0.6 - abs(elev - 0.6))
                    terrain[y][x] = self.FOREST if self.rng.random() < p_forest else self.PLAIN

            # Rivers: carve them through plains/forests only, well away from mountains and sea
            mountain_cells = [(x, y) for y in range(h) for x in range(w) if terrain[y][x] in (self.MOUNTAIN, self.SNOW)]
            dist_to_mtn = self._distance_field(mountain_cells) if mountain_cells else [[10**9] * w for _ in range(h)]

            # Choose river sources: higher land, far from sea, and not close to mountains (to avoid mountain-water adjacency via beaches)
            candidates = []
            for y in range(h):
                for x in range(w):
                    if terrain[y][x] in (self.PLAIN, self.FOREST):
                        if dist_to_sea[y][x] >= 8 and dist_to_mtn[y][x] >= 3 and height[y][x] >= 0.62:
                            candidates.append((height[y][x], x, y))
            candidates.sort(reverse=True)

            river_count = 2 + int(self.rng.random() * 4)  # 2..5 rivers
            self.rng.shuffle(candidates)
            sources = []
            # Pick diverse sources (avoid clustering)
            taken = set()
            for _, x, y in candidates:
                if len(sources) >= river_count:
                    break
                ok = True
                for sx, sy in sources:
                    if abs(sx - x) + abs(sy - y) < 30:
                        ok = False
                        break
                if ok and (x, y) not in taken:
                    sources.append((x, y))
                    taken.add((x, y))

            sea_codes = {self.SEA_DEEP, self.SEA_SHALLOW}
            forbidden_for_river = {self.MOUNTAIN, self.SNOW, self.SEA_DEEP, self.SEA_SHALLOW}

            # Carve rivers; if carving fails, we keep going (map can have fewer rivers)
            for s in sources:
                self._carve_river(
                    terrain=terrain,
                    height=height,
                    start=s,
                    sea_codes=sea_codes,
                    forbidden_codes=forbidden_for_river,
                )

            # Apply beaches around all water (sea + rivers)
            self._apply_beaches(terrain)

            # Recompute shallow vs deep sea after beaches/rivers (sea colors remain, but ensure shore is shallow)
            for y in range(h):
                for x in range(w):
                    if terrain[y][x] == self.SEA_DEEP:
                        for nx, ny in self._neighbors4(x, y):
                            if self._in_bounds(nx, ny, w, h) and terrain[ny][nx] == self.BEACH:
                                terrain[y][x] = self.SEA_SHALLOW
                                break

            # Final validation; if invalid, retry
            if self._validate(terrain):
                return terrain

        # If we couldn't validate after retries, return the last generated terrain (still drawn),
        # but it may violate constraints in rare cases.
        return terrain


class MapApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Biome Map Generator (240x160)")

        self.gen = WorldMapGenerator()

        self.canvas = tk.Canvas(
            self.root,
            width=WorldMapGenerator.W * WorldMapGenerator.SCALE,
            height=WorldMapGenerator.H * WorldMapGenerator.SCALE,
            highlightthickness=0,
            bd=0
        )
        self.canvas.pack()

        self.btn = tk.Button(self.root, text="Regenerate map", command=self.regenerate)
        self.btn.pack(pady=6)

        self._img_base = None
        self._img_scaled = None
        self._img_item = None

        self.regenerate()

    def _terrain_to_photoimage(self, terrain):
        w, h = WorldMapGenerator.W, WorldMapGenerator.H
        img = tk.PhotoImage(width=w, height=h)

        # Build rows as Tcl lists of colors for speed (160 rows only)
        for y in range(h):
            row_colors = []
            trow = terrain[y]
            for x in range(w):
                row_colors.append(WorldMapGenerator.CODE_TO_COLOR[trow[x]])
            # PhotoImage expects a string like "{#rrggbb #rrggbb ...}"
            img.put("{" + " ".join(row_colors) + "}", to=(0, y))
        return img

    def regenerate(self):
        terrain = self.gen.generate()

        # Create base image (240x160) and scale it up for clearer display
        self._img_base = self._terrain_to_photoimage(terrain)
        self._img_scaled = self._img_base.zoom(WorldMapGenerator.SCALE, WorldMapGenerator.SCALE)

        if self._img_item is None:
            self._img_item = self.canvas.create_image(0, 0, anchor="nw", image=self._img_scaled)
        else:
            self.canvas.itemconfig(self._img_item, image=self._img_scaled)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    t1 = time.perf_counter()
    app = MapApp()
    t2 = time.perf_counter()
    print(t2-t1)
    app.run()
    
