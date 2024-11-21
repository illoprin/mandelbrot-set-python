# Importing pygame libs
import pygame as pg
from pygame.locals import *

# Importing numba jit compiler
# from numba import njit, prange, jit

# importing taichi GPU accelerated compiler
import taichi as ti

# Importing math libs
import numpy as np

# Standard imports
import sys

# All constants
SCREEN_RES = (1280, 720)
width, height = SCREEN_RES
aspect = width/height

# LOAD GRADIENT TEXTURE
gradient = pg.image.load("gradient_01.jpg")
texture_size = max(gradient.get_size()) - 1
texture_array = pg.surfarray.array3d(gradient).astype(dtype=np.uint32)


def clamp(value, minimum, maximum):
    return min(max(minimum, value), maximum)

@ti.data_oriented
class MondelbrotSet:
    def __init__(self, app):
        self.app = app
        self.zoom: ti.uint64 = 1
        self.x: ti.float64 = -.3
        self.y: ti.float64 = .0

        ### Fractal Parametres ##
        self.max_iterations = 256
        self.area_x = 2.0
        self.area_y = 2.0
        #####################

        ti.init(arch=ti.cuda)

        self.pixel_array = ti.Vector.field(3, ti.uint32, (width, height))
        self.texture_field = ti.Vector.field(3, ti.uint32, (1, texture_size))
        self.texture_field.from_numpy(texture_array)

    ##### MAIN Mondelbrot Set Calculations #####
    # @staticmethod
    # @njit(fastmath=True, parallel=True)
    # def construct_cpua(screen_array, size=1, offset_x=0, offset_y=0, max_iterations=100, treshold=2) -> np.array:
    #     for x in prange(width):
    #         for y in prange(height):
    #             # standard coords
    #             a = (x / width) * 2 * 2 - 2
    #             b = (y / height) * 2 * 2 - 2
                
    #             ## rescale standard coords                
    #             a = a / size * aspect + offset_x
    #             b = b / size + offset_y
                
    #             ca = a
    #             cb = b
    #             iterations = 0
    #             while iterations < max_iterations:
    #                 aa = a*a - b*b
    #                 bb = 2 * a * b
    #                 a = aa + ca
    #                 b = bb + cb
    #                 if abs(a + b) > treshold:
    #                     break
    #                 iterations += 1
    #             # COLORING
    #             if iterations == max_iterations:
    #                 gray = 0
    #             else:
    #                 gray = int (texture_size * (iterations / max_iterations))
                
    #             screen_array[x, y] = texture_array[gray, 0]
    #     return screen_array
    ###############################################


    #### GPU Accelerated Mandelbrot Set Calculations ####
    @ti.kernel
    def construct_gpua(
        self, offset_x: ti.float64, offset_y: ti.float64, zoom: ti.float64, max_iter: ti.int64
    ):
        for x, y in self.pixel_array:
            c = ti.Vector(
                [
                ((x / width) * self.area_x * 2 - self.area_x) * aspect / zoom + offset_x,
                ((y / height) * self.area_y * 2 - self.area_y) / zoom + offset_y
                ], ti.float64
            )
            z = ti.Vector([0.0, 0.0], ti.float64)

            iterations = 0
            while iterations < max_iter:
                z = ti.Vector([z.x**2 - z.y**2 + c.x, 2*z.x*z.y + c.y])
                if z.dot(z) > 4:
                    break
                iterations += 1

            # u = (iterations - ti.math.log2(ti.math.log2(z.dot(z))) + 4.0) / float(max_iter) * texture_size
            u = (iterations / max_iter * texture_size)
            # u = ti.math.clamp(u, 1, texture_size)
            self.pixel_array[x, y] = self.texture_field[0, int(u)]

    # Expencive calculations
    def update(self):
        self.construct_gpua(self.x, self.y, self.zoom, self.max_iterations)
        print("Redraw Fractal Array")

    def render(self):
        pg.surfarray.blit_array(self.app.screen, self.pixel_array.to_numpy())


    def increase_zoom (self, zoom):
        if self.zoom + zoom > 0 and (self.zoom <= 23357547167344 or zoom < 0):
            self.zoom += zoom

    def add_pos (self, dx, dy, delta_time):
        self.x += dx/self.zoom*delta_time
        self.y += dy/self.zoom*delta_time

    def increase_max_iterations(self, val):
        self.max_iterations += val
        self.max_iterations = clamp(self.max_iterations, 8, 2048)

class App:
    def __init__(self):
        pg.init()
        self.screen = pg.display.set_mode(SCREEN_RES, pg.SCALED)
        self.is_running = True
        self.clock = pg.time.Clock()
        self.time = 0
        self.delta_time = 10
        self.velocity = 5
        self.need_redraw = True
        self.fractal = MondelbrotSet(self)
        
    def run(self):
        while self.is_running:
            self.handle_events()
            self.update()

            
            self.screen.fill((20, 20, 20))
            
            # WRITE RENDER CODE HERE
            self.render()
            self.clock.tick(0)
            
            pg.display.set_caption(f'FPS: {self.clock.get_fps():.0f}')
            pg.display.flip()
        self.exit()

    def exit(self):
        pg.quit()
        sys.exit()

    def handle_events(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.is_running = False
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_r:
                    self.fractal.max_iterations = 100
                    self.need_redraw = True
                if event.key == pg.K_ESCAPE:
                    self.is_running = False
                
        keys = pg.key.get_pressed()

        zoom = 0
        di = 0
        dx, dy = 0, 0
        if keys[pg.K_UP]:
            zoom = self.fractal.zoom * .05
        elif keys[pg.K_DOWN]:
            zoom = -self.fractal.zoom * .05

        if keys[pg.K_LEFT]:
            di -= 1
        elif keys[pg.K_RIGHT]:
            di += 1

        if keys[pg.K_w]:
            dy -= self.velocity
        elif keys[pg.K_s]:
            dy += self.velocity

        if keys[pg.K_a]:
            dx -= self.velocity
        elif keys[pg.K_d]:
            dx += self.velocity

        if zoom != 0 or dx != 0 or dy != 0 or di != 0:
            self.fractal.add_pos(dx, dy, self.delta_time)
            self.fractal.increase_zoom(zoom)
            self.fractal.increase_max_iterations(di)
            print(f'Zoom level is: {self.fractal.zoom:.0f}')
            self.need_redraw = True
        

    def render(self):
        self.fractal.render()

    def update(self):
        # Expencive calculations
        if self.need_redraw:
            self.time += 1
            self.delta_time = self.clock.get_time()/1000
            self.fractal.update()
            self.need_redraw = False


if __name__ == '__main__':
    app = App()
    app.run()
