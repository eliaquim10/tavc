from tensorflow.python.keras.models import Model
from tensorflow.python.keras.applications.vgg19 import VGG19
from .variaveis import *

import tensorflow as tf
import os

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

Loader_RGB_MEAN = np.array([0.4488, 0.4371, 0.4040]) * 255

def normalize(x, rgb_mean=Loader_RGB_MEAN):
    return (x - rgb_mean) / 127.5

def denormalize(x, rgb_mean=Loader_RGB_MEAN):
    return x * 127.5 + rgb_mean

def normalize_01(x):
    """Normalizes RGB images to [0, 1]."""
    return x / 255.0

def normalize_m11(x):
    """Normalizes RGB images to [-1, 1]."""
    return x / 127.5 - 1

def denormalize_m11(x):
    """Inverse of normalize_m11."""
    return (x + 1) * 127.5

def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)
    

# %% [markdown]
# #### Metricas

# %%
def psnr(x1, x2):
    return tf.image.psnr(x1, x2, max_val=255)

# %% [markdown]
# #### Transformacões

# %%
# @tf.function()
def load(image_file):
    # Read and decode an image file to a uint8 tensor
    try:
        image = tf.io.read_file(image_file)
        image = tf.io.decode_jpeg(image, 3)
    except:
        print(image_file)
        return None
    return image

def find_bound_box(filename):
    import xml.etree.ElementTree as ET
    #parsing XML file

    tree = ET.parse(filename)
    #fetching the root element

    root = tree.getroot()
    bound_box = []
    for object in root.findall('object'):
        name_element = object.find("name") 
        bndbox = object.find("bndbox") 
        xmin, ymin, xmax, ymax = bndbox.find("xmin").text, bndbox.find("ymin").text, bndbox.find("xmax").text, bndbox.find("ymax").text
        name = name_element.text    

        # print(xmin, ymin, xmax, ymax)
        bound_box.append({"name": name,  "bbox":[int(xmin), int(ymin), int(xmax), int(ymax)]})

        # print(subelem.tag, object.attrib, subelem.text)

    return bound_box


def load_image(path):
    image = np.array(Image.open(path))
    return tf.cast(image, tf.float32)

# @tf.function()
def noiser(img, std):

    new_img = tf.cast(img, tf.float32)
    new_img = new_img / 255

    noise = tf.random.normal(shape = new_img.get_shape(), mean = 0.0, stddev = std, dtype = tf.float32) 
    noise_img = new_img + noise
    noise_img = tf.clip_by_value(noise_img, 0.0, 1.0)
    return noise_img, new_img

def noiser_np(filename):
    img = np.array(Image.open(filename))
    noise = np.random.normal(127, 255, img.get_shape())
    noise_img = img + noise
    noise_img = tf.clip_by_value(noise_img, 0., 255.)

    return noise_img, img

# @tf.function()
def random_crop(noise_img, denoise_img, scale=2):
    denoise_img_shape = noise_img.get_shape()[:2]
    
    noise_crop_size = [256, 256]
    # noise_crop_size[1] =  denoise_img_shape[1] // scale 
    # noise_crop_size[0] =  denoise_img_shape[0] // scale
    # print(noise_crop_size)

    denoise_w = tf.random.uniform(shape=(), maxval=denoise_img_shape[1] - noise_crop_size[1] + 1, dtype=tf.int32)
    denoise_h = tf.random.uniform(shape=(), maxval=denoise_img_shape[0] - noise_crop_size[0] + 1, dtype=tf.int32)

    noise_img_cropped = noise_img[denoise_h:denoise_h + noise_crop_size[0], denoise_w:denoise_w + noise_crop_size[1]]
    denoise_img_cropped = denoise_img[denoise_h:denoise_h + noise_crop_size[0], denoise_w:denoise_w + noise_crop_size[1]]

    return noise_img_cropped, denoise_img_cropped 

@tf.function()
def random_flip(denoise_img, noise_img):
    rn = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(rn < 0.5,
                   lambda: (denoise_img, noise_img),
                   lambda: (tf.image.flip_left_right(denoise_img),
                            tf.image.flip_left_right(noise_img)))

@tf.function()
def random_rotate(denoise_img, noise_img):
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(denoise_img, rn), tf.image.rot90(noise_img, rn)


# %%
def resolve_single(model, noise):
    return resolve(model, tf.expand_dims(noise, axis=0))[0]


def resolve(model, noise_batch):
    noise_batch = tf.cast(noise_batch, tf.float32)
    denoise_batch = model(noise_batch)
    denoise_batch = tf.clip_by_value(denoise_batch, 0, 255)
    denoise_batch = tf.round(denoise_batch)
    denoise_batch = tf.cast(denoise_batch, tf.uint8)
    return denoise_batch

def cast(denoise_batch):
    denoise_batch = tf.clip_by_value(denoise_batch, 0, 255)
    denoise_batch = tf.round(denoise_batch)
    denoise_batch = tf.cast(denoise_batch, tf.uint8)
    return denoise_batch

def resolve_mae(model, noise_batch):
    noise_batch = tf.cast(noise_batch, tf.float32)
    denoise_batch = model(noise_batch)
    denoise_batch = tf.clip_by_value(denoise_batch, 0, 255)
    denoise_batch = tf.round(denoise_batch)
    return denoise_batch


def evaluate(model, dataset):
    psnr_values = []
    for noise, real in dataset:
        denoise = resolve(model, noise)

        psnr_value = psnr(real, denoise)[0]
        psnr_values.append(psnr_value)
    return tf.reduce_mean(psnr_values)

def evaluate_mae(model, dataset):
    mae_values = []
    for noise, real in dataset:
        denoise = resolve_mae(model, noise)

        mae = metric_mae(real, denoise)
        mae_values.append(mae)
    return tf.reduce_mean(mae_values)

weights_dir = 'weights/srgan'
weights_file = lambda filename: os.path.join(weights_dir, filename)

os.makedirs(weights_dir, exist_ok=True)

def resolve_single(model, denoise):
    return resolve(model, tf.expand_dims(denoise, axis=0))[0]
    
# def load_image(path):
#     return np.array(Image.open(path))


def plot_sample(denoise, sr):
    plt.figure(figsize=(20, 10))

    images = [denoise, sr]
    titles = ['denoise', f'SR (x{sr.shape[0] // denoise.shape[0]})']

    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 2, i+1)
        plt.imshow(img)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])

def vgg_22():
    return _vgg(5)


def vgg_54():
    return _vgg(20)


def _vgg(output_layer):
    vgg = VGG19(input_shape=(None, None, 3), include_top=False)
    return Model(vgg.input, vgg.layers[output_layer].output)

def __main__():
    pass
if __name__ == "__main__":
    pass