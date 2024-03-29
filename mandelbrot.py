import tensorflow as tf
import numpy as np

import PIL.Image


def draw_fractal(a, fmt='jpeg'):
    """Display an array of iteration counts as a
    colorful picture of a fractal"""
    a_cyclic = (6.28 * a / 20.0).reshape(list(a.shape) + [1])
    img = np.concatenate([
        10 + 20 * np.cos(a_cyclic), 30 + 50 * np.sin(a_cyclic),
        155 - 80 * np.cos(a_cyclic)
    ], 2)

    img[a == a.max()] = 0
    a = img
    a = np.uint8(np.clip(a, 0, 255))

    with open('tmp/mandelbrot.jpg', 'w') as f:
        PIL.Image.fromarray(a).save(f, fmt)


# main
if __name__ == '__main__':
    sess = tf.InteractiveSession()

    # Use NumPy to create a 2D array of complex numbers and freely mix them
    # with TensorFlow
    y, x = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]
    z = x + 1j * y

    xs = tf.constant(z.astype(np.complex64))
    zs = tf.Variable(xs)
    ns = tf.Variable(tf.zeros_like(xs, tf.float32))

    # TensorFlow REQUIRES explicitely initializing variables before using them
    tf.global_variables_initializer().run()

    # Compute the new value z = z^2 + x
    zs_ = zs * zs + xs

    # Diverged with this new value?
    not_diverged = tf.abs(zs_) < 4

    # Operation to update the zs and the iteration count
    #
    # NOTE: We keep compute zs after they diverge! This is
    #       very wasteful! They are better, if a little less
    #       simple, way to do this.
    step = tf.group(
        zs.assign(zs_), ns.assign_add(tf.cast(not_diverged, tf.float32)))

    # ... and run it for a couple hundred steps
    for i in range(200):
        step.run()

    draw_fractal(ns.eval())
