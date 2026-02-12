import random
import tkinter as tk
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable


@dataclass(frozen=True)
class Colors:
    MOUNTAIN: str = "#7f7f7f"   # gray
    SNOW: str = "#ffffff"       # white
    PLAINS: str = "#9be28a"     # light green
    FOREST: str = "#1f6f2a"     # dark green
    BEACH: str = "#f1e27a"      # yellow
    SHALLOW: str = "#79c7ff"    # light blue
    DEEP: str = "#0b3d91"       # dark blue
    RIVER: str = "#79c7ff"      # same as shallow (per rules)


# Biome codes (ints for speed)
DEEP_SEA = 0
SHALLOW_SEA = 1
RIVER = 2
BEACH = 3
PLAINS = 4
FOREST = 5
MOUNTAIN = 6
SNOW = 7


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def neighbors4(x: int, y: int, n: int) -> Iterable[Tuple[int, int]]:
    if x > 0:
        yield x - 1, y
    if x + 1 < n:
        yield x + 1, y
    if y > 0:
        yield x, y - 1
    if y + 1 < n:
        yield x, y + 1


def box_blur(grid: List[List[float]], passes: int = 1) -> List[List[float]]:
    """Simple box blur with edge handling by clamping indices (fast + robust)."""
    n = len(grid)
    if n == 0:
        return grid
    m = len(grid[0])
    out = [row[:] for row in grid]

    for _ in range(max(0, passes)):
        tmp = [[0.0] * m for _ in range(n)]
        for y in range(n):
            y0 = max(0, y - 1)
            y1 = min(n - 1, y + 1)
            for x in range(m):
                x0 = max(0, x - 1)
                x1 = min(m - 1, x + 1)
                s = 0.0
                cnt = 0
                for yy in range(y0, y1 + 1):
                    row = out[yy]
                    for xx in range(x0, x1 + 1):
                        s += row[xx]
                        cnt += 1
                tmp[y][x] = s / cnt
        out = tmp
    return out


def make_value_noise(n: int, rng: random.Random) -> List[List[float]]:
    return [[rng.random() for _ in range(n)] for _ in range(n)]


def normalize(grid: List[List[float]]) -> List[List[float]]:
    n = len(grid)
    if n == 0:
        return grid
    mn = min(min(row) for row in grid)
    mx = max(max(row) for row in grid)
    if mx <= mn + 1e-12:
        return [[0.0 for _ in range(n)] for _ in range(n)]
    scale = 1.0 / (mx - mn)
    return [[(v - mn) * scale for v in row] for row in grid]


def radial_falloff(n: int, power: float = 2.5) -> List[List[float]]:
    """Higher in the center, lower near edges, range ~[0,1]."""
    c = (n - 1) / 2.0
    maxd = (2 * (c ** 2)) ** 0.5  # corner distance
    out = [[0.0] * n for _ in range(n)]
    for y in range(n):
        dy = y - c
        for x in range(n):
            dx = x - c
            d = (dx * dx + dy * dy) ** 0.5 / maxd
            # 1 in center, 0 at corners; sharpen with power
            out[y][x] = clamp((1.0 - d) ** power, 0.0, 1.0)
    return out


def combine_height(n: int, rng: random.Random) -> List[List[float]]:
    """
    Heightmap: multi-scale blurred noise + radial falloff.
    Produces continents with water around edges.
    """
    a = normalize(box_blur(make_value_noise(n, rng), passes=5))
    b = normalize(box_blur(make_value_noise(n, rng), passes=12))
    c = normalize(box_blur(make_value_noise(n, rng), passes=2))
    rf = radial_falloff(n, power=2.7)

    h = [[0.0] * n for _ in range(n)]
    for y in range(n):
        for x in range(n):
            # Weighted noise, then shape with falloff
            v = 0.55 * a[y][x] + 0.30 * b[y][x] + 0.15 * c[y][x]
            v = 0.15 + 0.85 * v  # keep a baseline
            v = v * (0.35 + 0.65 * rf[y][x])
            h[y][x] = v
    return normalize(h)


def combine_moisture(n: int, rng: random.Random) -> List[List[float]]:
    m1 = normalize(box_blur(make_value_noise(n, rng), passes=6))
    m2 = normalize(box_blur(make_value_noise(n, rng), passes=1))
    m = [[0.0] * n for _ in range(n)]
    for y in range(n):
        for x in range(n):
            m[y][x] = clamp(0.7 * m1[y][x] + 0.3 * m2[y][x], 0.0, 1.0)
    return normalize(m)


def classify_initial(
    height: List[List[float]],
    moisture: List[List[float]],
    sea_level: float,
    shallow_margin: float,
    mountain_level: float,
    snow_level: float,
) -> List[List[int]]:
    n = len(height)
    bio = [[PLAINS] * n for _ in range(n)]
    for y in range(n):
        for x in range(n):
            h = height[y][x]
            if h < sea_level:
                # deeper if significantly below sea level
                if h < sea_level - shallow_margin:
                    bio[y][x] = DEEP_SEA
                else:
                    bio[y][x] = SHALLOW_SEA
            else:
                if h >= snow_level:
                    bio[y][x] = SNOW
                elif h >= mountain_level:
                    bio[y][x] = MOUNTAIN
                else:
                    bio[y][x] = FOREST if moisture[y][x] > 0.55 else PLAINS
    return bio


def is_water(b: int) -> bool:
    return b in (DEEP_SEA, SHALLOW_SEA, RIVER)


def is_land(b: int) -> bool:
    return not is_water(b)


def pick_river_sources(
    height: List[List[float]],
    bio: List[List[int]],
    rng: random.Random,
    max_sources: int = 12,
    min_height: float = 0.72,
) -> List[Tuple[int, int]]:
    """Pick a few high points on land as river sources."""
    n = len(height)
    candidates: List[Tuple[float, int, int]] = []
    for y in range(n):
        for x in range(n):
            if bio[y][x] in (MOUNTAIN, SNOW) and height[y][x] >= min_height:
                candidates.append((height[y][x], x, y))

    if not candidates:
        return []

    # Take top fraction and sample from it
    candidates.sort(reverse=True)
    top = candidates[: max(10, len(candidates) // 6)]
    rng.shuffle(top)

    sources: List[Tuple[int, int]] = []
    used = set()
    for _, x, y in top:
        if (x, y) in used:
            continue
        sources.append((x, y))
        used.add((x, y))
        if len(sources) >= max_sources:
            break
    return sources


def trace_river(
    height: List[List[float]],
    bio: List[List[int]],
    start: Tuple[int, int],
    max_steps: int = 5000,
) -> List[Tuple[int, int]]:
    """
    Follow steepest descent until reaching sea.
    Robust: stops on loops/plateaus/step limit.
    """
    n = len(height)
    x, y = start
    path: List[Tuple[int, int]] = []
    visited = set()

    for _ in range(max_steps):
        if (x, y) in visited:
            break
        visited.add((x, y))
        path.append((x, y))

        # Stop if already adjacent to sea (not river) to ensure it can "end" at coast
        for nx, ny in neighbors4(x, y, n):
            if bio[ny][nx] in (DEEP_SEA, SHALLOW_SEA):
                path.append((nx, ny))
                return path

        # Choose neighbor with lowest height (ties: first found)
        best = None
        best_h = height[y][x]
        for nx, ny in neighbors4(x, y, n):
            nh = height[ny][nx]
            if nh < best_h:
                best_h = nh
                best = (nx, ny)

        if best is None:
            # plateau / local minimum: stop
            break
        x, y = best

        # If we stepped into sea, river ends
        if bio[y][x] in (DEEP_SEA, SHALLOW_SEA):
            return path

    return path


def carve_rivers(
    height: List[List[float]],
    bio: List[List[int]],
    rng: random.Random,
) -> None:
    sources = pick_river_sources(height, bio, rng)
    n = len(height)

    for s in sources:
        path = trace_river(height, bio, s)
        if len(path) < 10:
            continue

        # Mark river cells only on land; allow the final sea cell to remain sea
        for (x, y) in path:
            if 0 <= x < n and 0 <= y < n and is_land(bio[y][x]):
                bio[y][x] = RIVER


def add_beaches_around_water(bio: List[List[int]]) -> None:
    """Enforce: sea/rivers may only touch beaches (orthogonally)."""
    n = len(bio)
    to_beach = set()

    for y in range(n):
        for x in range(n):
            if is_water(bio[y][x]):
                for nx, ny in neighbors4(x, y, n):
                    if not is_water(bio[ny][nx]):
                        to_beach.add((nx, ny))

    # Convert all marked neighbors to beach (even if they were mountains/snow; fixed later)
    for x, y in to_beach:
        bio[y][x] = BEACH


def cleanup_beaches(bio: List[List[int]], moisture: List[List[float]]) -> None:
    """Ensure beaches always touch water, and act as land bridge between water and (plains/forest)."""
    n = len(bio)
    for y in range(n):
        for x in range(n):
            if bio[y][x] != BEACH:
                continue
            touches_water = any(is_water(bio[ny][nx]) for nx, ny in neighbors4(x, y, n))
            if not touches_water:
                # Stray beach -> turn into plains/forest
                bio[y][x] = FOREST if moisture[y][x] > 0.55 else PLAINS


def enforce_mountain_rules(bio: List[List[int]], moisture: List[List[float]]) -> None:
    """
    - Mountain ranges may only touch plains or forests (and mountains/snow internally).
      So they must NOT touch beaches or water.
    - Higher altitudes (snow) must be surrounded by mountain ranges (mountain or snow).
    """
    n = len(bio)

    # 1) Snow must be surrounded by mountains/snow
    for y in range(n):
        for x in range(n):
            if bio[y][x] != SNOW:
                continue
            ok = True
            for nx, ny in neighbors4(x, y, n):
                if bio[ny][nx] not in (MOUNTAIN, SNOW):
                    ok = False
                    break
            if not ok:
                bio[y][x] = MOUNTAIN

    # 2) Mountains cannot touch beach/water; convert offending mountains to plains/forest
    for y in range(n):
        for x in range(n):
            if bio[y][x] not in (MOUNTAIN, SNOW):
                continue
            bad = False
            for nx, ny in neighbors4(x, y, n):
                if bio[ny][nx] in (BEACH, DEEP_SEA, SHALLOW_SEA, RIVER):
                    bad = True
                    break
            if bad:
                bio[y][x] = FOREST if moisture[y][x] > 0.55 else PLAINS


def stabilize_constraints(
    bio: List[List[int]],
    moisture: List[List[float]],
    iterations: int = 5,
) -> None:
    """
    Iterate constraint enforcement a few times to settle interactions:
    beaches added -> mountains adjusted -> beaches cleaned -> etc.
    """
    for _ in range(max(1, iterations)):
        add_beaches_around_water(bio)
        enforce_mountain_rules(bio, moisture)
        cleanup_beaches(bio, moisture)


class MapGeneratorApp:
    def __init__(self, root: tk.Tk, pixel_size: int = 520, cell_size: int = 4):
        if pixel_size <= 0 or cell_size <= 0 or pixel_size % cell_size != 0:
            raise ValueError("pixel_size must be positive and divisible by cell_size")

        self.root = root
        self.colors = Colors()
        self.pixel_size = pixel_size
        self.cell_size = cell_size
        self.n = pixel_size // cell_size

        root.title("Biome Map Generator (Tkinter)")

        self.canvas = tk.Canvas(root, width=pixel_size, height=pixel_size, highlightthickness=0)
        self.canvas.grid(row=0, column=0, columnspan=3, padx=8, pady=8)

        self.seed_var = tk.StringVar(value=str(random.randrange(1_000_000_000)))
        tk.Label(root, text="Seed:").grid(row=1, column=0, sticky="e", padx=6, pady=6)
        self.seed_entry = tk.Entry(root, textvariable=self.seed_var, width=18)
        self.seed_entry.grid(row=1, column=1, sticky="w", padx=6, pady=6)

        tk.Button(root, text="Generate", command=self.generate).grid(row=1, column=2, padx=6, pady=6)

        self.legend = tk.Label(
            root,
            text=(
                "Colors: mountains=gray, snow=white, plains=light green, forests=dark green,\n"
                "beaches=yellow, rivers/shallow sea=light blue, deep sea=dark blue."
            ),
            justify="left",
        )
        self.legend.grid(row=2, column=0, columnspan=3, sticky="w", padx=8, pady=(0, 8))

        self.bio: Optional[List[List[int]]] = None
        self.generate()

    def biome_color(self, b: int) -> str:
        c = self.colors
        if b == DEEP_SEA:
            return c.DEEP
        if b == SHALLOW_SEA:
            return c.SHALLOW
        if b == RIVER:
            return c.RIVER
        if b == BEACH:
            return c.BEACH
        if b == PLAINS:
            return c.PLAINS
        if b == FOREST:
            return c.FOREST
        if b == MOUNTAIN:
            return c.MOUNTAIN
        if b == SNOW:
            return c.SNOW
        return "#ff00ff"  # should not happen

    def generate(self) -> None:
        # Robust seed parsing
        s = self.seed_var.get().strip()
        try:
            seed = int(s)
        except ValueError:
            seed = random.randrange(1_000_000_000)
            self.seed_var.set(str(seed))

        rng = random.Random(seed)

        # Tunable thresholds (kept conservative to make constraints easier to satisfy)
        sea_level = 0.46
        shallow_margin = 0.10
        mountain_level = 0.72
        snow_level = 0.86

        height = combine_height(self.n, rng)
        moisture = combine_moisture(self.n, rng)

        bio = classify_initial(height, moisture, sea_level, shallow_margin, mountain_level, snow_level)
        carve_rivers(height, bio, rng)
        stabilize_constraints(bio, moisture, iterations=6)

        # Final pass: ensure no water touches non-beach (safety net)
        add_beaches_around_water(bio)
        cleanup_beaches(bio, moisture)
        enforce_mountain_rules(bio, moisture)

        self.bio = bio
        self.draw()

    def draw(self) -> None:
        if self.bio is None:
            return

        self.canvas.delete("all")
        n = self.n
        cs = self.cell_size

        # Draw as rectangles (fast enough for ~17k cells).
        # For larger n, consider a PhotoImage-based renderer.
        for y in range(n):
            y0 = y * cs
            y1 = y0 + cs
            row = self.bio[y]
            for x in range(n):
                x0 = x * cs
                x1 = x0 + cs
                self.canvas.create_rectangle(
                    x0, y0, x1, y1,
                    outline="",
                    fill=self.biome_color(row[x]),
                )


def main() -> None:
    root = tk.Tk()
    MapGeneratorApp(root, pixel_size=520, cell_size=4)  # 130x130 logical grid
    root.mainloop()


if __name__ == "__main__":
    main()
