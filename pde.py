import tensorflow as tf
import numpy as np
import PIL.Image
from io import BytesIO


def draw_array(a, fname, fmt='jpeg', range=[0, 1]):
    """Draw an array as a image file"""
    a = (a - range[0]) / float(range[1] - range[0]) * 255
    a = np.uint8(np.clip(a, 0, 255))

    with open(fname, 'w') as f:
        PIL.Image.fromarray(a).save(f, fmt)


sess = tf.InteractiveSession()


def make_kernel(a):
    """Transform a 2D array into a convolution kernel"""
    a = np.asarray(a)
    a = a.reshape(list(a.shape) + [1, 1])
    return tf.constant(a, dtype=1)


def simple_conv(x, k):
    """A simplified 2D convolution operation"""
    x = tf.expand_dims(tf.expand_dims(x, 0), -1)
    y = tf.nn.depthwise_conv2d(x, k, [1, 1, 1, 1], padding='SAME')
    return y[0, :, :, 0]


def laplace(x):
    """Compute the 2D laplacian of an array"""
    laplace_k = make_kernel([[0.5, 1.0, 0.5], [1.0, -6., 1.0][0.5, 1.0, 0.5]])
    return simple_conv(x, laplace_k)
