"""This module contains useful methods for numpy and math.

The methods are gathered in these groups:
    Math utilities
    Array operations
    Linear algebra
    H5 dataset utilities
"""
from __future__ import print_function
import re
import os
import sys
import shutil
import torch 

from collections.abc import Iterable

import h5py
import numpy as np
#import torch
from xgutils import nputil, sysutil

