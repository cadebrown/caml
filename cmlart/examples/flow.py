#!/usr/bin/env python3
""" cmlart/examples/flow.py - Optical flow

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
parser.add_argument("--feedback", type=float, help="hallucination/artifact feedback. used with optical flow to create more concise animations", default=0.0)
parser.add_argument("--steps", type=int, help="steps for dreaming", default=100)
parser.add_argument("--tile-size", type=int, help="tile size for random rolling", default=1024)
parser.add_argument("--octaves", type=int, help="number of 'octaves' the image goes through, gradually upsizing", default=0)
parser.add_argument("--octave-scale", type=float, help="the scale between each octave (between 1 and 2 is normally good)", default=1.5)
parser.add_argument("--cut-at", type=int, help="how many frames to cut at", default=None)


args = parser.parse_args()


import cmlart
import tensorflow as tf
import numpy as np
import cv2


# Dream model
dream = cmlart.dreamutil.make_IV3(args.layers)

# Helper function to run an image through the dream and get the result, substituting arguments
def run_dream(img):
    return dream(img, args.rate, args.steps, args.tile_size, args.octaves, args.octave_scale)

# Start off with no hallucination array
hallu = None


# Output video
out = cmlart.VideoWriter(args.output)


for i, img in enumerate(cmlart.VideoReader(args.input)):
    if args.cut_at is not None and i >= args.cut_at:
        break

    if args.size is not None:
        img = cv2.resize(img, args.size)
        #img = cmlart.resize(img, args.size)

    # Convert to float32
    img = np.float32(img)

    # Convert to grayscale, for use in optical flow
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w, d = img.shape

    print("on frame %i" % (i,))

    if hallu is None or args.feedback == 0:
        # Initialize for the first time (or, if no feedback, we can do it everytime)
        hallu = run_dream(img)
    else:
        # Calculate optical flow, which is a map of how each pixel moves
        flow = cv2.calcOpticalFlowFarneback(img_prev_gray, img_gray, None,
            pyr_scale=0.5, 
            levels=4, 
            winsize=32, 
            iterations=3,
            poly_n=7,
            poly_sigma=1.5,
            flags=0
        )

        # Now, re-scale the flow
        flow = flow * -(1.0)

        # Adjust flow so that offsets are absolute instead of relative
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:,np.newaxis]

        # Get just the hallucinations (i.e. the halluciations minus the source image, which
        #   gives just the artifacts). This is basically the previous gradient times the learning rate
        hallu_diff = hallu - img_prev

        # Re-map the difference so we 'shift' the hallucinations with the motion of the video
        hallu_diff = cv2.remap(hallu_diff.numpy(), flow, None, cv2.INTER_LINEAR).astype(np.float32)
        print (hallu_diff)

        hallu = img + args.feedback * hallu_diff
        hallu = run_dream(hallu)


        # Perform deep dream on the image plus the difference added
        #hallu = run_dream((1.0 - args.feedback) * img + args.feedback * hallu_diff)

    cv2.imshow("hallu", hallu.numpy())
    cv2.waitKey(1)

    out.add(hallu)

    img_prev = img
    img_prev_gray = img_gray

out.writer.release()
