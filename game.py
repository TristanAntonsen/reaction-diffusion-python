import pygame
import numpy as np
import pandas as pd
import cv2

NEIGHBORS = [(-1, -1),(0, -1),(1, -1),(1, 0),(1, 1),(0, 1),(-1, 1),(-1, 0)]
KERNEL = np.array([[0.05, 0.2, 0.05], [0.2, -1, 0.2], [0.05, 0.2, 0.05]])
BOX = np.ones((3, 3), np.float32)/9

WADJ = 0.2
WDIAG = 0.05
WSELF = -1

DA=1.
DB=.5
f=.055; k=.062 # original
# f=.025; k=.052
# f=.0367; k=.0649 # mitosis
# k=.05452; f=.10159
# k=.06362; f=.06625

class Grid:
    def __init__(self, screen, gridsize) -> None:
        self.gridsize = gridsize
        self.width: int = int(np.ceil(screen.get_width() / self.gridsize))
        self.height: int = int(np.ceil(screen.get_height() / self.gridsize))
        self.a: np.ndarray = np.ones(self.width * self.height).reshape(self.width, self.height)
        self.b: np.ndarray = np.zeros(self.width * self.height).reshape(self.width, self.height)
        self.c: np.ndarray = np.ones(self.width * self.height).reshape(self.width, self.height)
        self.g: np.ndarray = np.zeros(self.width * self.height).reshape(self.width, self.height)
        self.f: float = f
        self.k: float = k

    def get_tile_indices(self, pos) -> (int, int):
        i: int = int(np.floor(pos[0] / self.gridsize))
        j: int = int(np.floor(pos[1] / self.gridsize))

        return (i, j)
    
    def get(self, i, j):
        
        return (self.a[i, j], self.b[i, j])

    def set(self, i, j, val):

        self.a[i, j] = val[0]
        self.b[i, j] = val[1]


    # def draw(self, screen: pygame.surface.Surface) -> None:

    #     for i in range(0, self.width):
    #         for j in range(0, self.height):
    #             c = self.c[i, j]
    #             rect = pygame.Rect(i * self.gridsize, j * self.gridsize, self.gridsize, self.gridsize)
    #             pygame.draw.rect(screen, (c,c,c), rect)

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

        last_a = self.a.copy()
        self.a = next_a
        self.b = next_b

        # color array
        self.c = np.floor((self.a - self.b) * 255.)
        gradient = np.gradient(self.c)
        self.dc = (gradient[0] - gradient[1]) - 50
        self.c = np.clip(self.c, 0., 255.)
    
    def seed_block(self, i, j, width, height, a, b):

        self.a[i:i + width,j:j + height] = a
        self.b[i:i + width,j:j + height] = b
        self.c = np.floor((self.a - self.b) * 255.)
        self.c = np.clip(self.c, 0, 255.)

def gray(grid):
    c = 255 * (grid.c / grid.c.max())
    w, h = c.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] = c
    ret[:, :, 1] = c + grid.g * 100
    ret[:, :, 0] = c
    return ret


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", default="False")
    parser.add_argument("-s", "--step", default="False")

    args = parser.parse_args()

    debug = args.debug.lower() == "true"
    step = args.step.lower() == "true"

    xres = 480
    yres = 720
    block_size = 5
    pygame.init()
    screen = pygame.display.set_mode((xres, yres))
    pygame.display.set_caption('Reaction Diffusion')
    clock = pygame.time.Clock()

    running = True

    grid = Grid(screen, 1)

    iterate = True
    for n in range(100):
        x = np.random.randint(0, grid.width)
        y = np.random.randint(0, grid.height)
        block_size = np.random.randint(10, 20)
        grid.seed_block(x, y, block_size, block_size, 0, 1)
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        surf = pygame.surfarray.make_surface(gray(grid))

        screen.blit(surf, (0, 0))

        events = pygame.event.get()

        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    iterate = True

        if step:
            for event in events:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        grid.iteration()

        if iterate and not step:
            grid.iteration()
            clock.tick()  # limit FPS
            if debug:
                print(clock.get_fps())

        mouse_pressed = pygame.mouse.get_pressed()

        if mouse_pressed[0]:
            pos = pygame.mouse.get_pos()
            tile = grid.get_tile_indices(pos)
            grid.seed_block(*tile, block_size, block_size, 0, 1)


        if mouse_pressed[2]:
            pos = pygame.mouse.get_pos()
            tile = grid.get_tile_indices(pos)
            grid.seed_block(*tile, block_size, block_size, 1, 0)

        pygame.display.update()

    pygame.quit()


if __name__ == "__main__":
    main()