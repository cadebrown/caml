#!/usr/bin/env python3
""" cmlart/dream.py - Commandline utility for 'dreaming' images


You can run with `python3 -mcmlart.dream [args...]`

For example:

```
$ python3 -mcmlart.dream img/firepat.png out.png --steps 256 --rate 0.01 -s 1024 1024 --layers mixed1 --octaves 3 --octave-scale 2.25

```


You should export your PYTHONPATH to include this directory. For example, if you cloned the repo in `~/projects`,
  add `export PYTHONPATH="$PYTHONPATH:$HOME/projects/cmlart"` to your .bashrc

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

# Now, replace metadata tag in name (if applicable)
# NOTE: Metadata has the format: (steps, rate, octaves, octave_scale)
args.output = args.output.replace(':META:', 's%i_r%.5f_o%i_os%.2f_%s' % (args.steps, args.rate, args.octaves, args.octave_scale, '_'.join(args.layers)))

# Import library
import cmlart as cml
import cmlart.dreamutil as du

import tensorflow as tf

# Get a base classification model to optimize
model_base = du.base_IV3(args.layers)

# Create dreaming model
model_dream = du.DreamModel(model_base)

# Helper function to run an image and get the result
def run(img):
    # Preprocess image 
    img = du.preproc_IV3(img)

    # Actually apply dream model to it
    img = model_dream(img, args.rate, args.steps, args.tile_size, args.octaves, 1.5)

    # De-process it back into color space
    img = du.deproc_IV3(img)
    return img

# Now, read the input image
img = cml.imread(args.input)

# Resize input, if given the `-s` switch
if args.size:
    img = cml.resize(img, args.size)

st = time.time()

# Process the image
img = run(img)

et = time.time() - st
print("took %2.3fs" % (et,))

# Save the image
cml.imwrite(img, args.output)
