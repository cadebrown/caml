#!/usr/bin/env python3
""" cmlart/examples/dreamanim.py 


@author: Cade Brown <cade@cade.site>
"""

import argparse
import time

parser = argparse.ArgumentParser(description='Dream-ify an image')

parser.add_argument('input', help='input image to dreamify')
parser.add_argument('output', help='output image location (NOTE: "[[META]]" is replaced with a metadatastring)')

parser.add_argument("-s", "--size", nargs=2, type=int, help="transform size", default=None)
parser.add_argument("--layers", type=str, nargs='+', help="steps for dreaming", default=['mixed3', 'mixed5'])
parser.add_argument("--rate", type=float, help="image delta speed (higher means more changes, lower means less)", default=0.01)
parser.add_argument("--steps", type=int, help="steps for dreaming", default=100)
parser.add_argument("--tile-size", type=int, help="tile size for random rolling", default=1024)
parser.add_argument("--octaves", type=int, help="number of 'octaves' the image goes through, gradually upsizing", default=0)
parser.add_argument("--octave-scale", type=float, help="the scale between each octave (between 1 and 2 is normally good)", default=1.5)

args = parser.parse_args()

import tensorflow as tf
import cmlart
import cv2

# Dream model
dream = cmlart.dreamutil.make_IV3(args.layers)

# Helper function to run an image through the dream and get the result, substituting arguments
def run_dream(img):
    return dream(img, args.rate, args.steps, args.tile_size, args.octaves, args.octave_scale)

# Now, read the input image
img = cmlart.imread(args.input)

# Resize input, if given the `-s` switch
if args.size is not None:
    img = cmlart.resize(img, args.size)
  
else:
    args.size = img.shape[:2]

st = time.time()

fps = 24.0

zoom_per_sec = 1.4

# Output video
out = cmlart.VideoWriter(args.output, fps)

for i in range(24 * 16):
    t = i / fps

    print("on frame %i" % (i,))

    # Size change
    sizefac = zoom_per_sec ** (-1 / fps)
    hh = int((1 - sizefac) * args.size[0] // 2)
    ww = int((1 - sizefac) * args.size[1] // 2)

    # Crop to center, upscale
    img = cmlart.resize(img[hh:-hh, ww:-ww, ...], args.size)

    # Now, run the image out
    img = run_dream(img)
    out.add(img)


    cv2.imshow("img", img.numpy())
    cv2.waitKey(1)



out.writer.release()
