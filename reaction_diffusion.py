import pygame
import numpy as np
import cv2

""" Antonsen 2024 """

# ignore divide by zero errors
np.seterr(divide='ignore', invalid='ignore')

NEIGHBORS = [(-1, -1),(0, -1),(1, -1),(1, 0),(1, 1),(0, 1),(-1, 1),(-1, 0)]
KERNEL = np.array([[0.05, 0.2, 0.05], [0.2, -1, 0.2], [0.05, 0.2, 0.05]])
BOX = np.ones((3, 3), np.float32)/9

WADJ = 0.2
WDIAG = 0.05
WSELF = -1

DA=1.
DB=.5

STATES = {
    0 : [],
    1 : [[2,3]],
    2 : [[1,2]],
    3 : [[1,3]],
    4 : [[0,1]],
    5 : [[0,3], [1, 2]],
    6 : [[0,2]],
    7 : [[0,3]],
    8 : [[0,3]],
    9 : [[0,2]],
    10 : [[0,1], [2, 3]],
    11 : [[0,1]],
    12 : [[1,3]],
    13 : [[1,2]],
    14 : [[2,3]],
    15 : []
}

INSTRUCTIONS = """
INSTRUCTIONS:
- Click to draw
- Right click to erase
- Space bar to run
- "S" to export"""

def get_state(a,b,c,d,iso):
    
    # < iso = 0, > iso = 1
    sa = (a - iso) > 0
    sb = (b - iso) > 0
    sc = (c - iso) > 0
    sd = (d - iso) > 0

    # Index for lookup table
    s = round(sa * 8 + sb * 4 + sc * 2 + sd * 1)

    return s

def lerp(a, b, t):
    c = a + t * (b - a)
    return c

def lerp_points(p0, p1, t):

    return [
        lerp(p0[0], p1[0], t),
        lerp(p0[1], p1[1], t)
    ]

def find_lerp_factor(v0, v1, iso_val):
    t = (iso_val - v0) / (v1 - v0)
    return max(min(1, t), 0)

class Grid:
    def __init__(self, screen, gridsize) -> None:
        self.gridsize = gridsize
        self.width: int = int(np.ceil(screen.get_width() / self.gridsize))
        self.height: int = int(np.ceil(screen.get_height() / self.gridsize))
        self.a: np.ndarray = np.ones(self.width * self.height).reshape(self.width, self.height)
        self.b: np.ndarray = np.zeros(self.width * self.height).reshape(self.width, self.height)
        self.c: np.ndarray = np.ones(self.width * self.height).reshape(self.width, self.height)
        self.g: np.ndarray = np.zeros(self.width * self.height).reshape(self.width, self.height)
        self.f: float = 0.055
        self.k: float = 0.062

    def get_tile_indices(self, pos) -> (int, int):
        i: int = int(np.floor(pos[0] / self.gridsize))
        j: int = int(np.floor(pos[1] / self.gridsize))

        return (i, j)
    
    def get(self, i, j):
        
        return (self.a[i, j], self.b[i, j])

    def getc(self, i, j):
        
        return self.c[i, j]

    def set(self, i, j, val):

        self.a[i, j] = val[0]
        self.b[i, j] = val[1]
        
    def iteration(self):
        next_a: np.ndarray = np.zeros(self.width * self.height).reshape(self.width, self.height)
        next_b: np.ndarray = np.zeros(self.width * self.height).reshape(self.width, self.height)

        # convolutions
        convA = cv2.filter2D(src=self.a, ddepth=-1, kernel=KERNEL) 
        convB = cv2.filter2D(src=self.b, ddepth=-1, kernel=KERNEL) 

        # reaction
        ab2 = self.a * self.b * self.b

        # next a & b
        next_a = self.a + ((DA * convA) - (ab2) + (self.f * (1 - self.a)))
        next_b = self.b + ((DB * convB) + (ab2) - ((self.k + self.f) * self.b))

        self.a = next_a
        self.b = next_b

        # color array
        self.c = (self.a - self.b)
        gradient = np.gradient(self.c)
        self.g = np.clip(gradient[1] - gradient[0], 0., 1.)
        self.c = np.clip(self.c, 0., 1.)
    
    def seed_block(self, i, j, width, height, a, b):
        self.a[i:i + width,j:j + height] = a
        self.b[i:i + width,j:j + height] = b
        self.c = np.floor((self.a - self.b) * 255.)
        self.c = np.clip(self.c, 0, 255.)

    def write_csv(self, path):
        np.savetxt(path, self.c, delimiter=",")

    def write_img(self, path):
        np.savetxt(path, self.c, delimiter=",")


def seed_random(grid, n, block_size):
    for _ in range(n):
        x = np.random.randint(0, grid.width)
        y = np.random.randint(0, grid.height)
        grid.seed_block(x, y, block_size, block_size, 0, 1)


def march(grid: Grid, iso, interpolated=True):
    # marching squares to extract edges
    graph = []

    for i in range(grid.height - 1):
        for j in range(grid.width - 1):
            
            scale = 1
            x = i * scale
            y = j * scale

            ## values at (corner) points

            v0 = grid.getc(j    , i    )
            v1 = grid.getc(j    , i + 1)
            v2 = grid.getc(j + 1, i + 1)
            v3 = grid.getc(j + 1, i    )

            ## Interopolation factors
            ta = find_lerp_factor(v0, v1, iso)
            tb = find_lerp_factor(v1, v2, iso)
            tc = find_lerp_factor(v3, v2, iso) # flip due to sign change
            td = find_lerp_factor(v0, v3, iso)

            ## edge point locations (interpolated)
            a = [x + ta * scale , y              ] # 01
            b = [x + scale      , y + tb * scale ] # 12
            c = [x + tc * scale , y + scale      ] # 23
            d = [x              , y + td * scale ] # 30

            ## edge point locations (not interpolated)
            _a = [x + 0.5 * scale , y               ] # 01
            _b = [x + scale       , y + 0.5 * scale ] # 12
            _c = [x + 0.5 * scale , y + scale       ] # 23
            _d = [x               , y + 0.5 * scale ] # 30

            edge_points = [a,b,c,d]

            if interpolated == False:
                edge_points = [_a,_b,_c,_d]

            state = get_state(v0, v1, v2, v3, iso)

            edges = STATES[state]
            
            for line in edges:
                p1 = edge_points[line[0]]
                p2 = edge_points[line[1]]
                graph.append((p1, p2))
    return graph

def color(grid):
    c = (grid.c / grid.c.max()) * 255.

    w, h = c.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)

    ret[:, :, 0] = c # r
    ret[:, :, 2] = c # g
    ret[:, :, 1] = c # b

    return ret

def export_svg(grid, export_path, scale=2):

    import svgwrite
    dwg = svgwrite.Drawing(export_path)
    stroke = svgwrite.rgb(0, 0, 0, '%')

    edges = march(grid, 0.5, True)

    for line in edges:
        p1 = line[0]
        p2 = line[1]
        x1 = p1[1] * scale # reversed x & y for consistency
        y1 = p1[0] * scale
        x2 = p2[1] * scale
        y2 = p2[0] * scale
        dwg.add(dwg.line((x1, y1), (x2, y2), stroke=stroke))
    
    dwg.save(export_path)

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", default="False")
    args = parser.parse_args()
    debug = args.debug.lower() == "true"

    print(INSTRUCTIONS)

    # Main parameters
    res = (240, 240)
    substeps = 20
    block_size = 50

    # sample reaction/diffusion values
    f = 0.055
    k = 0.062
    f=.0367; k=.0649 # mitosis
    # f=.025; k=.055
    # k=.05452; f=.10159
    # k=.06362; f=.06625

    pygame.init()
    screen = pygame.display.set_mode(res)
    pygame.display.set_caption('Reaction Diffusion')
    clock = pygame.time.Clock()

    running = True
    iterate = False

    grid = Grid(screen, 1)
    grid.k = k
    grid.f = f

    # seed_random(grid, 100, 5)

    while running:

        # mouse and key events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    iterate = not iterate
                elif event.key == pygame.K_s:
                    grid.write_csv("reaction_diffusion.csv")
                    export_svg(grid, "reaction_diffusion.svg", scale=2)
                    pygame.image.save(screen, "reaction_diffusion.png")
                    running = False
                    print("SAVE")

        mouse_pressed = pygame.mouse.get_pressed()
    
        if mouse_pressed[0]:
            pos = pygame.mouse.get_pos()
            tile = grid.get_tile_indices(pos)
            grid.seed_block(*tile, block_size, block_size, 0, 1)

        if mouse_pressed[2]:
            pos = pygame.mouse.get_pos()
            tile = grid.get_tile_indices(pos)
            grid.seed_block(*tile, block_size, block_size, 1, 0)

        # drawing the pixels
        surf = pygame.surfarray.make_surface(color(grid))
        screen.blit(surf, (0, 0))

        if iterate:
            for _ in range(substeps):
                grid.iteration()
            clock.tick()
            if debug:
                print(clock.get_fps())

        pygame.display.update()

    pygame.quit()


if __name__ == "__main__":
    main()