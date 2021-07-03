""" cmlart/__init__.py - Cade's Machine Learning ART utilities


@author: Cade Brown <cade@cade.site>
"""

from . import dreamutil

import tensorflow as tf
import numpy as np
import PIL
import cv2

### Image Utilities ###

# Loads an image from a filename
def imread(fname):
    img = PIL.Image.open(fname).convert('RGB')
    return np.array(img)

# Saves an image to a filename
def imwrite(img, fname):
    PIL.Image.fromarray(np.array(img)).save(fname)

def resize(img, sz):
    return tf.image.resize(img, sz)



### Video Utilities ###

class VideoWriter:

    def __init__(self, fname, fps=24.0, size=None):
        self.fname = fname
        self.fps = fps
        self.size = size
        self.writer = None

    # Add output image
    def add(self, img):
        if self.writer is None:
            if self.size is None:
                self.size = img.shape[:2]

            self.writer = cv2.VideoWriter(self.fname, None, self.fps, self.size)

        # Actually write image
        self.writer.write(np.array(img))
