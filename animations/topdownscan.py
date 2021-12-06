from datetime import datetime

import numpy as np
import math

from tree_animator import TreeAnimator
from utils.color import hsv_to_rgb

NUM_BANDS = 10
ANIMATION_PERIOD = 3
# matplotlib doesn't work on the raspberry pi.
# hardcoding the generated colormap
"""
import numpy as np
import matplotlib.pyplot as plt

colormap = plt.get_cmap("winter")
print("[")
for i in np.linspace(0,1,5):
    print("  {},".format(colormap(i)[0:3]))
print("]")
"""
COLORMAP = [
  (0.0, 0.0, 1.0),
  (0.0, 0.25098039215686274, 0.8745098039215686),
  (0.0, 0.5019607843137255, 0.7490196078431373),
  (0.0, 0.7529411764705882, 0.6235294117647059),
  (0.0, 1.0, 0.5),
]

class TopDownScanAnimator(TreeAnimator):
    def initialize_animation(self):
        # This function only gets called once at the start of the animation.
        # Do anything in here that is related to the setup of your animation, initializing variables, connecting to APIs, whatever
        coords = self._xyz_coords.copy()
        # z is the height               
        self.min_height = coords[:,2].min()                      
        self.max_height = coords[:,2].max()
        amp_height = self.max_height - self.min_height
        self.band_height = amp_height / NUM_BANDS

        print("Height (min/max/amp):", self.min_height, "/", self.max_height, "/", amp_height)
        print("Band height:", self.band_height)

    def calculate_colors(self, xyz_coords, start_time):
        # this function gets called every few milliseconds, and it's purpose is to return the colors we want each light to be.

        # the arguments passed into this function are xyz_coords. this is an Nx3 array, where N is the number of lights (500)
        # start_time is the datetime straight after the initialize_animation() is ran
        # we can calculate how many seconds the animation has been running for by:
        num_seconds_since_start = (datetime.now() - start_time).total_seconds()
        # each time this function is called, num_seconds_since_start will get larger, and this can be used to control and time animations

        # This function is expected to return colors. colors are in RGB and in the range 0-255, for example black is [0,0,0], white is [255,255,255], red is [255,0,0]
        colors = np.full((self.NUM_LIGHTS, 3), fill_value=0, dtype=np.uint8)
        num_seconds_since_start = (datetime.now() - start_time).total_seconds()

        print("num_seconds_since_start: ", num_seconds_since_start)
        color_i = math.floor(num_seconds_since_start / ANIMATION_PERIOD)
        color_rgb = self.get_band_color(color_i)
        band_i = math.floor((num_seconds_since_start / ANIMATION_PERIOD) % 1.0 * NUM_BANDS)
        min_band_height, max_band_height = self.get_band_bounds(band_i)

        print("color: ", color_i, "/", color_rgb)
        print("min_band_height/max_band_height:", min_band_height, "/", max_band_height)
        colors[np.logical_and(xyz_coords[:,2] >= min_band_height, xyz_coords[:,2] < max_band_height)] = color_rgb


        return colors

    def get_band_bounds(self, i):
        if i < 0 or i >= NUM_BANDS:
            raise "out of bounds band index: {}".format(i)
        
        # go from top to bottom of the tree.
        max_band_height = self.max_height - i * self.band_height
        return (max_band_height - self.band_height, max_band_height)
        

    def get_band_color(self, i):
        num_colors = len(COLORMAP)
        cmap = COLORMAP[i % num_colors]
        return (cmap[0]*255, cmap[1]*255, cmap[2]*255)



if __name__ == "__main__":
    anim = TopDownScanAnimator()

    anim.animation_loop()