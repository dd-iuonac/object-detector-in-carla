import colorsys
import math
from random import random

import numpy as np


def rand_color(seed):
    """Return random color based on a seed"""
    random.seed(seed)
    col = colorsys.hls_to_rgb(random.random(), random.uniform(.2, .8), 1.0)
    return int(col[0] * 255), int(col[1] * 255), int(col[2] * 255)


def vector3d_to_array(vec3d):
    return np.array([vec3d.x, vec3d.y, vec3d.z])


def degrees_to_radians(degrees):
    return degrees * math.pi / 180
