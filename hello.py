import tensorflow as tf
hello = tf.constant('Hello, TensorFlow (' + tf.__version__ + ')!')
sess = tf.Session()
print(sess.run(hello))
a = tf.constant(10)
b = tf.constant(32)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(a + b))
