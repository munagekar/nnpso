'''
Custom TensorFlow Layers
Author : Abhishek Munagekar
'''
import tensorflow as tf


# Activation Function
def activate(input_layer, act='relu', name='activation'):
    if act is None:
        return input_layer
    if act == 'relu':
        return tf.nn.relu(input_layer, name)
    if act == 'sqr':
        return tf.square(input_layer, name)
    if act == 'sqr_sigmoid':
        return tf.nn.sigmoid(tf.square(input_layer, name))
    if act == 'sigmoid':
        return tf.nn.sigmoid(input_layer, name)


# Fully connected custom layer for PSO
# Supported activation function types : None,relu,sqr,sqr_sigmoid,sigmoid
def fc(input_tensor, n_output_units, scope,
       activation_fn='relu', uniform=False):
    shape = [input_tensor.get_shape().as_list()[-1], n_output_units]
    # Use the Scope specified
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Init Weights
        if uniform:
            weights = tf.Variable(tf.random_uniform(
                shape=shape,
                dtype=tf.float32,
                minval=-10,
                maxval=10),
                name='weights')
        else:
            weights = tf.Variable(tf.truncated_normal(
                shape=shape,
                mean=0.0,
                stddev=0.1,
                dtype=tf.float32),
                name='weights')
        # Init Biases
        biases = tf.Variable(
            tf.zeros(shape=[n_output_units]), name='biases', dtype=tf.float32)
        # Particle Best
        pbest_w = tf.get_variable(
            shape=shape,
            name='pbest_w',
            initializer=tf.random_uniform_initializer)
        pbest_b = tf.get_variable(
            shape=[n_output_units],
            name='pbest_b',
            initializer=tf.random_uniform_initializer)

        # Velocities
        vel_weights = tf.Variable(tf.random_uniform(
            shape=shape,
            dtype=tf.float32,
            minval=-0.001,
            maxval=0.001),
            name='vel_weights')
        vel_biases = tf.Variable(tf.random_uniform(
            shape=[n_output_units],
            dtype=tf.float32,
            minval=-0.001,
            maxval=0.001),
            name='vel_biases')

        # Perform actual feedforward
        act = tf.matmul(input_tensor, weights) + biases
        pso_tupple = [weights, biases,
                      pbest_w, pbest_b,
                      vel_weights, vel_biases]
        # Activate And Return
        return activate(act, activation_fn), pso_tupple


# Magnitude Clipper
# Magmax can be either a Tensor or a Float
def maxclip(tensor, magmax):
    # assertion commented out to allow usage of both Tensor & Integer
    # assert magmax > 0, "magmax argument in maxclip must be positive"
    return tf.minimum(tf.maximum(tensor, -magmax), magmax)
