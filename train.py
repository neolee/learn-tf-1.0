import numpy as np
import tensorflow as tf

# Path for TensorFlow event logs, use the following line to see in TensorBoard
# tensorboard --logdir /tmp/tensorflow_logs/example
logs_path = '/tmp/tensorflow_logs/example'

# Model input and output
with tf.name_scope('input'):
    # None -> batch size can be any size, 784 -> flattened mnist image
    x = tf.placeholder(tf.float32, name="x-input")
    # target 10 output classes
    y = tf.placeholder(tf.float32, name="y-input")

# Model parameters will change during training so we use tf.Variable
with tf.name_scope("weights"):
    w = tf.Variable([.3], tf.float32)

# Model bias
with tf.name_scope("biases"):
    b = tf.Variable([-.3], tf.float32)

# Model
with tf.name_scope('Model'):
    linear_model = w * x + b
with tf.name_scope('Loss'):
    loss = tf.reduce_sum(tf.square(linear_model - y))
with tf.name_scope('SGD'):
    optimizer = tf.train.GradientDescentOptimizer(0.01)

# Trainer data & loop
train = optimizer.minimize(loss)

x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)  # reset values

summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})

# Evaluate training accuracy
curr_w, curr_b, curr_loss = sess.run([w, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s" % (curr_w, curr_b, curr_loss))
