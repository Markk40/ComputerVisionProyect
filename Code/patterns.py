import os
import cv2
import imageio
import numpy as np
from typing import List
from utils import non_max_suppression, get_hsv_color_ranges
import glob

def load_images(filenames: List) -> List:
    return [cv2.imread(filename) for filename in filenames]

imgs_path = [f for f in glob.glob("../data/*.jpg")]
imgs = load_images(imgs_path)