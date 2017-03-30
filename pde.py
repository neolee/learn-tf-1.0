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
    laplace_k = make_kernel([[0.5, 1.0, 0.5], [1.0, -6., 1.0], [0.5, 1.0, 0.5]])
    return simple_conv(x, laplace_k)


# Init states
N = 500

u_init = np.zeros([N, N], dtype=np.float32)
ut_init = np.zeros([N, N], dtype=np.float32)

for n in range(40):
  a,b = np.random.randint(0, N, 2)
  u_init[a,b] = np.random.uniform()

draw_array(u_init, 'tmp/pde-start.jpg', range=[-0.1, 0.1])

# Parameters:
# eps -- time resolution
# damping -- wave damping
eps = tf.placeholder(tf.float32, shape=())
damping = tf.placeholder(tf.float32, shape=())

# Create variables for simulation state
u  = tf.Variable(u_init)
ut = tf.Variable(ut_init)

# Discretized PDE update rules
u_ = u + eps * ut
ut_ = ut + eps * (laplace(u) - damping * ut)

# Operation to update the state
step = tf.group(
  u.assign(u_),
  ut.assign(ut_))

# Initialize state to initial conditions
tf.global_variables_initializer().run()

# Run 1000 steps of PDE
for i in range(1000):
  # Step simulation
  step.run({eps: 0.03, damping: 0.04})
  draw_array(u.eval(), 'tmp/pde.jpg', range=[-0.1, 0.1])