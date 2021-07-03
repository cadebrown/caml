""" cmlart/dreamutil/__init__.py - Deep-dream like implementations


@author: Cade Brown <cade@cade.site>
"""


import tensorflow as tf
import numpy as np

import cmlart as cml


""" Returns a base model that can be used with `DreamModel`, taken from the InceptionV3 network

  layernames: List of layers used as outputs to the base model (these are what are going to be optimized).
                See the file `info/IV3_layers.txt` to see valid values
"""
def base_IV3(layernames=['mixed1']):
    # Base model 
    model_class = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    
    # To print out all layers
    #for layer in model_class.layers:
    #    print (" -", layer.name)

    # Return a wrapping model that only outputs the given layer names
    return tf.keras.Model(
        inputs=model_class.input,
        outputs=[model_class.get_layer(name).output for name in layernames]
    )

""" Pre-processes an image for InceptionV3 network, returns a tensor that can be used

"""
def preproc_IV3(img):
    return tf.convert_to_tensor(
        tf.keras.applications.inception_v3.preprocess_input(img)
    )

""" De-processes an image from InceptionV3 network, returns an image that can be viewed

"""
def deproc_IV3(img):
    return tf.cast(255 * (img + 1.0) / 2.0, tf.uint8)


""" Returns a base model that can be used with `DreamModel`, taken from the DenseNet201 network

"""
def base_DN201(layernames=['mixed1']):
    # Base model 
    model_class = tf.keras.applications.DenseNet201(include_top=False, weights='imagenet')
    
    # To print out all layers
    for layer in model_class.layers:
        print (" -", layer.name)

    # Return a wrapping model that only outputs the given layer names
    return tf.keras.Model(
        inputs=model_class.input,
        outputs=[model_class.get_layer(name).output for name in layernames]
    )


""" Pre-processes an image for DenseNet201 network, returns a tensor that can be used

"""
def preproc_DN201(img):
    return tf.convert_to_tensor(
        tf.keras.applications.densenet.preprocess_input(img)
    )

""" De-processes an image from DenseNet201 network, returns an image that can be viewed

"""
def deproc_DN201(img):
    return tf.cast(255 * img, tf.uint8)
    #return tf.cast(255 * (img + 1.0) / 2.0, tf.uint8)



""" DreamModel - Model for creating 'dreamy' or 'deep-dream' images from an internal model

Typically, you should create with 1 argument, which is the internal model. You can get
one based on the InceptionV3 network (a great default) by calling the `base_IV3()` method

"""
class DreamModel(tf.Module):

    # Initialize with the internal model, which is typically a classification model
    def __init__(self, model):
        self.model = model

    # Calculates the loss for a single image
    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None,None,3], dtype=tf.float32),
        )
    )
    def calc_loss(self, img):
        # Turn into an array of 1
        img = tf.expand_dims(img, axis=0)

        # Sum up each activation
        loss = tf.zeros(())
        for act in self.model(img):
            loss += tf.math.reduce_mean(act)

        # Now, see our total loss
        return loss

    # Utility function to calculate a random roll of an image
    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None,None,3], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.int32),
        )
    )
    def calc_randroll(self, img, tile_size):
        # Calculate random shift
        shift = tf.random.uniform(shape=[2], minval=-tile_size, maxval=tile_size, dtype=tf.int32)
        img_rolled = tf.roll(img, shift=shift, axis=[0, 1])
        return shift, img_rolled

    # Function call operator used to transfrom an image, returns loss and the new image
    @tf.function(
        input_signature=(
            tf.TensorSpec(shape=[None,None,3], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.float32),
            tf.TensorSpec(shape=[], dtype=tf.int32),
            tf.TensorSpec(shape=[], dtype=tf.int32),
            tf.TensorSpec(shape=[], dtype=tf.int32),
            tf.TensorSpec(shape=[], dtype=tf.float32),
        )
    )
    def __call__(self, img, rate=0.01, steps=100, tile_size=256, octaves=0, octave_scale=1.5):
        # Capture original shape, for octave computation
        #origimg = img
        origshape = tf.cast(tf.shape(img)[:-1], tf.float32)

        # Iterate over the octave indices (default, just iterates with octave=0)
        for octave in tf.range(-octaves, 1):

            # Now, compute the shape that this octave would be
            # With octave=0, this is just the original shape. Otherwise, it's smaller
            shape = tf.cast(
                origshape * octave_scale ** tf.cast(octave, tf.float32),
                tf.int32
            )
            # Resize the image to this shape. The first time through, this will downsize
            #   to the smallest octave. Other times, it will size up to the next
            img = tf.image.resize(img, shape)

            # Iterate over the number of steps
            # TODO: Should this be variable depending on the octave?
            for step in range(steps):
                # Calculate a random roll, to displace 
                shift, img_rolled = self.calc_randroll(img, tile_size)

                # Gradients start at 0
                grads = tf.zeros_like(img_rolled)

                # Skip the last tile, unless there's only one tile.
                xs = tf.range(0, shape[0], tile_size)[:-1]
                ys = tf.range(0, shape[1], tile_size)[:-1]

                # If there was no tile at all, just do one anyway
                if not tf.cast(len(xs), bool):
                    xs = tf.constant([0])
                if not tf.cast(len(ys), bool):
                    ys = tf.constant([0])
                
                # Now, iterate over tile coordinates
                for x in xs:
                    for y in ys:
                        # Calculate the gradients for this tile.
                        with tf.GradientTape() as tape:
                            # Calculate gradients relative to the rolled image
                            tape.watch(img_rolled)
                    
                            # Extract the tile
                            tile = img_rolled[x:x+tile_size, y:y+tile_size]

                            # Calculate loss for that tile
                            loss = self.calc_loss(tile)

                        # Update the gradients
                        grads = grads + tape.gradient(loss, img_rolled)


                # Undo the random shift applied to the image and gradients
                grads = tf.roll(grads, shift=-shift, axis=[0, 1])

                # Normalize the gradients, avoiding division by 0
                grads /= tf.math.reduce_std(grads) + 1e-8 

                # Apply to image
                img = img + grads * rate
                
                # Clip to valid range
                img = tf.clip_by_value(img, -1, 1)
                #img = tf.clip_by_value(img, 0, 1)

        # Return final image result
        return img
