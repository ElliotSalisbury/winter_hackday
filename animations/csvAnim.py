from datetime import datetime

import numpy as np

from tree_animator import TreeAnimator
from utils.color import hsv_to_rgb


class csvAnim(TreeAnimator):
    #duration = .01
    def initialize_animation(self):
        # This function only gets called once at the start of the animation.
        # Do anything in here that is related to the setup of your animation, initializing variables, connecting to APIs, whatever
        self.colors = np.zeros((self.NUM_LIGHTS, 3), dtype=np.uint8)
        self.lastTime = -1.0
        self.currentFrame = 0
        pass

    def calculate_colors(self, xyz_coords, start_time):
        # this function gets called every few milliseconds, and it's purpose is to return the colors we want each light to be.

        # the arguments passed into this function are xyz_coords. this is an Nx3 array, where N is the number of lights (500)
        # start_time is the datetime straight after the initialize_animation() is ran
        # we can calculate how many seconds the animation has been running for by:
        num_seconds_since_start = (datetime.now() - start_time).total_seconds()
        # each time this function is called, num_seconds_since_start will get larger, and this can be used to control and time animations
        if (num_seconds_since_start - self.lastTime) >= .01:
            self.lastTime = num_seconds_since_start
            self.currentFrame = (self.currentFrame + 1)%600
            filename = "csvAnim/" + str(self.currentFrame) + ".csv"
            f = open(filename, "r")
            for line in f.readlines():
                LC = line.split(",")
                if int(LC[0]) <= self.NUM_LIGHTS:
                    self.colors[int(LC[0])] = np.array([int(LC[1]), int(LC[2]), int(LC[3])])

        # This function is expected to return colors. colors are in RGB and in the range 0-255, for example black is [0,0,0], white is [255,255,255], red is [255,0,0]
        return self.colors

if __name__ == "__main__":
    anim = csvAnim()

    anim.animation_loop()