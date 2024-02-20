import os
import sys
import scipy
import numpy as np
import matplotlib.pyplot as plt
from xgutils import *

import igl

ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))+"/assets/"
def load_mesh(name="Utah.obj"):
    face, vert = igl.read_triangle_mesh(os.path.join(ASSETS_DIR, name) )
    return face, vert

