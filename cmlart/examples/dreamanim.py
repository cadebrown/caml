#!/usr/bin/env python3
""" cmlart/examples/dreamanim.py 


@author: Cade Brown <cade@cade.site>
"""

import argparse
import time
import os

parser = argparse.ArgumentParser(description='Dream-ify an image')

parser.add_argument('input', help='input image to dreamify')
parser.add_argument('output', help='output image location (NOTE: ":META:" is replaced with a metadatastring)')

parser.add_argument("-s", "--size", nargs=2, type=int, help="transform size", default=None)
parser.add_argument("--layers", type=str, nargs='+', help="steps for dreaming", default=['mixed3', 'mixed5'])
parser.add_argument("--rate", type=float, help="image delta speed (higher means more changes, lower means less)", default=0.01)
parser.add_argument("--steps", type=int, help="steps for dreaming", default=100)
parser.add_argument("--tile-size", type=int, help="tile size for random rolling", default=1024)
parser.add_argument("--octaves", type=int, help="number of 'octaves' the image goes through, gradually upsizing", default=0)
parser.add_argument("--octave-scale", type=float, help="the scale between each octave (between 1 and 2 is normally good)", default=1.5)
parser.add_argument('--preview', action='store_true', help="If given, preview the current frame")
parser.add_argument('--fps', type=float, default=30.0, help="Frames per second of the footage")
parser.add_argument('--len', type=float, default=15.0, help="Length of the output, in seconds")
parser.add_argument('--zoom-rate', type=float, default=1.8, help="The rate it zooms, per second. For example, 2.0 sooms in 2x per second")
#parser.add_argument('--start-frame', type=int, default=0, help="The frame to start computing on")

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

fps = args.fps
#fps = 8.0
dur = args.len
zoom_per_sec = args.zoom_rate

# Output video
#out = cmlart.VideoWriter(args.output, fps)
try:
    os.mkdir(args.output[:args.output.rfind('/')])
except:
    pass

print ("-- STARTING --")

for i in range(int(fps * dur)):
    t = i / fps
    outname = args.output.replace(':META:', '%05i' % (i,))

    print("on frame %i/%i" % (i, int(fps * dur)))

    # Size change
    sizefac = zoom_per_sec ** (-1 / fps)
    hh = int((1.0 - sizefac) * args.size[0] // 2)
    ww = int((1.0 - sizefac) * args.size[1] // 2)

    # Crop to center, upscale
    img = cmlart.resize(img[hh:-hh, ww:-ww, ...], args.size)

    print ("  dreaming...")
    # Now, run the image out
    img = run_dream(img)
    
    print ("  writing...")
    #out.add(img)
    cmlart.imwrite(img, outname)

    if args.preview:
        print ("  showing...")
        cv2.imshow("img", img.numpy())
        cv2.waitKey(1)



#out.writer.release()
