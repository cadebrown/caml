# cmlart

Cade's Machine Learning ART (CMLART) is a collection of utilities for doing AI-assisted (or, as I have started calling it, human-assisted) art. It contains some implementations of basic algorithms such as deep dream (based on InceptionNet), as well as some GAN-based approaches

## setup

To install requirements, run `pip3 install -r requirements.txt`

Also, sometimes you have to run something like `sudo apt install python3-opencv`

If you don't want to install this package, you can run export the path to `PYTHONPATH`. For example, if you cloned the repo in `~/projects`, add `export PYTHONPATH="$PYTHONPATH:$HOME/projects/cmlart"` to your shell (for example, in your `~/.bashrc`)




## examples

```shell
# create a video
$ python3 -mcmlart.examples.dream img/ddd.jpg out_:META:.webp --steps 64 --rate 0.0025 -s 512 512 --layers mixed3 --octaves 2 --octave-scale 1.4
```

```shell
$ python3 -mcmlart.examples.dreamanim img/firepat_1080p.png SDM_END.mp4 -s 1080 1920 --layers mixed8 --rate 0.012 --steps 128 --octaves 3 --octave-scale 2.25 --preview # fps=8#
```

### create video from images (ffmpeg)

```shell
# example assuming `out0/` contains the images
$ ffmpeg -framerate 15 -i out0/%05d.webp -pix_fmt yuv420p -c:v libx264 out.mp4
```

## links

  * https://www.tensorflow.org/install/gpu - for installing GPU libs
  * https://github.com/NVIDIA/flownet2-pytorch
  * https://github.com/nerdyrodent/VQGAN-CLIP

