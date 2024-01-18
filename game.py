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
#f=.0367; k=.0649 # mitosis
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

def color(grid):
    c = (grid.c / grid.c.max()) * 255.
    g = (grid.g / grid.g.max())

    w, h = c.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)

    # r
    ret[:, :, 0] = c
    # b
    ret[:, :, 2] = c
    # g
    ret[:, :, 1] = c

    return ret

# def gray(grid):
#     c = 255 * (grid.c / grid.c.max())
#     w, h = c.shape
#     ret = np.empty((w, h, 3), dtype=np.uint8)
#     ret[:, :, 2] = c
#     ret[:, :, 1] = c #+ grid.g * 100
#     ret[:, :, 0] = c
#     return ret


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", default="False")

    args = parser.parse_args()

    debug = args.debug.lower() == "true"

    res = (512, 512)
    substeps = 10
    block_size = 5
    pygame.init()
    screen = pygame.display.set_mode(res)
    pygame.display.set_caption('Reaction Diffusion')
    clock = pygame.time.Clock()

    running = True

    grid = Grid(screen, 1)

    # seed blocks
    # for _ in range(10):
    #     x = np.random.randint(0, grid.width)
    #     y = np.random.randint(0, grid.height)
    #     grid.seed_block(x, y, block_size, block_size, 0, 1)

    iterate = False

    while running:

        # mouse and key events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    iterate = True
                elif event.key == pygame.K_s:
                    grid.write_csv("domain.csv")
                    pygame.image.save(screen, "domain.png")
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
            clock.tick()  # limit FPS
            if debug:
                print(clock.get_fps())

        pygame.display.update()

    pygame.quit()


if __name__ == "__main__":
    main()