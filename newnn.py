'''
This file contains the new network definition
Author: Abhishek Munagekar
Language: Python 3
'''
# The number of values for xor


import itertools
from functools import reduce
import operator
import tensorflow as tf
import time
import random
import layers


# Dataset Generation for xor

N_IN = 5
N_BATCHSIZE = 32


# Parameters for pso
# TODO: Add batch size and other properties
N_PARTICLES = 1
N_ITERATIONS = int(3)
P_BEST_FACTOR = 2
G_BEST_FACTOR = 2
# Velocity Decay specifies the multiplier for the velocity update
VELOCITY_DECAY = 0.5
t_VELOCITY_DECAY = tf.constant(value=VELOCITY_DECAY,
                               dtype=tf.float32,
                               name='vel_decay')


# Basic Neural Network Definition
# Simple feedforward Network

HIDDEN_LAYERS = [3, 2]
LAYERS = [N_IN] + HIDDEN_LAYERS + [1]
print('The Network Structure is', LAYERS)


# Xorgenerator Function
def xor_next_batch(batch_size, n_input):
    batch_x = []
    batch_y = []
    for i in range(batch_size):
        x = []
        y = []
        ans = 0
        for j in range(n_input):
            x.append(random.randint(0, 1))
            ans ^= x[j]
        y.append(ans)
        batch_y.append(y)
        batch_x.append(x)
    return batch_x, batch_y


# A list of lists having N_IN elements all either 0 or 1
xor_in = [list(i) for i in itertools.product([0, 1], repeat=N_IN)]
# A list having 2^N lists each having xor of each input list in the list
# of lists
xor_out = list(map(lambda x: [(reduce(operator.xor, x))], xor_in))


net_in = tf.placeholder(dtype=tf.float32,
                        shape=[N_BATCHSIZE, N_IN],
                        name='net_in')

label = tf.placeholder(dtype=tf.float32,
                       shape=[N_BATCHSIZE, 1],
                       name='net_label')


print('Starting to Build Network')

# MULTI-PARTICLE NEURAL NETS

losses = []
nets = []
weights = []
biases = []
pweights = []
pbiases = []
vweights = []
vbiases = []
ovbiases = []
random_values = []
bias_updates = []
weight_updates = []
nextvbs = []

# Fixed Constant

# NOTE:Graph Isn't initialized Properly Needsd to be fixed
# TODO:Parellized the following loop
for pno in range(N_PARTICLES):
    net = net_in
    # Define the parameters
    w = None
    b = None
    pw = None
    pb = None
    vw = None
    vb = None
    gw = None
    gb = None
    for idx, num_neuron in enumerate(LAYERS[1:]):
        layer_scope = 'pno' + str(pno + 1) + 'fc' + str(idx + 1)
        net, w, b, pw, pb, pf, vw, vb = layers.fc(input_tensor=net,
                                                  n_output_units=num_neuron,
                                                  activation_fn='sigmoid',
                                                  scope=layer_scope,
                                                  uniform=True)
        vweights.append(vw)
        vbiases.append(vb)
        weights.append(w)
        biases.append(b)
        with tf.variable_scope(layer_scope, reuse=False):
            # Constants & Other Random Variables
            pbestrand = tf.Variable(tf.random_uniform(shape=[],
                                                      maxval=P_BEST_FACTOR),
                                    name='pbestrand')

            gbestrand = tf.Variable(tf.random_uniform(shape=[],
                                                      maxval=G_BEST_FACTOR),
                                    name='gbestrand')
            random_values.append(pbestrand)
            random_values.append(gbestrand)

        # Multiply by the Velocity Decay
        nextvw = tf.multiply(vw, t_VELOCITY_DECAY)
        nextvb = tf.multiply(vb, t_VELOCITY_DECAY)

        # Differences between Particle Best & Current
        pdiffw = tf.multiply(tf.subtract(pw, w), pbestrand)
        pdiffb = tf.multiply(tf.subtract(pb, b), pbestrand)
        # Define & Reuse the GBest
        with tf.variable_scope("gbest", reuse=tf.AUTO_REUSE):
            gw = tf.get_variable(name='fc' + str(idx + 1) + 'w',
                                 shape=[LAYERS[idx], LAYERS[idx + 1]],
                                 initializer=tf.zeros_initializer)

            gb = tf.get_variable(name='fc' + str(idx + 1) + 'b',
                                 shape=[LAYERS[idx + 1]],
                                 initializer=tf.zeros_initializer)

        # Differences between Global Best & Current
        gdiffw = tf.multiply(tf.subtract(gw, w), gbestrand)
        gdiffb = tf.multiply(tf.subtract(gb, b), gbestrand)
        vw = nextvw + pdiffw + gdiffw
        vb = nextvb + pdiffb + gdiffb


        weight_update = tf.assign(w, w + vw, validate_shape=True)
        weight_updates.append(weight_update)
        bias_update = tf.assign(b, b + vb, validate_shape=True)
        bias_updates.append(bias_update)

    # Define loss for each of the particle nets
    loss = tf.nn.l2_loss(net - label)
    # Update the lists
    nets.append(net)
    losses.append(loss)
    print('Building Net:', pno + 1)


print('Network Build Successful')


# Initialize the entire graph
init = tf.global_variables_initializer()
print('Graph Init Successful:')

for var in tf.global_variables():
    print(var)

req_list = weights
# Define the updates which are to be done before each iterations
random_updates = [r.initializer for r in random_values]
updates = weight_updates + bias_updates + random_updates

print('Hello')
for x, y in zip(vbiases, ovbiases):
    if x is y:
        print('Okay')
    else:
        print('Not Okay')

with tf.Session() as sess:
    sess.run(init)

    # Write The graph summary
    summary_writer = tf.summary.FileWriter(
        '/tmp/tf/logs', sess.graph_def)
    start_time = time.time()
    for i in range(N_ITERATIONS):
        # Reinitialize the Random Values at each iteration
        sess.run(updates)

        #xor_in,xor_out = xor_next_batch(N_BATCHSIZE,N_IN)
        dict_out = sess.run(req_list, feed_dict={
                            net_in: xor_in, label: xor_out})

        _losses = dict_out
        print('Losses:', _losses, 'Iteration:', i)

    end_time = time.time()
    # Close the writer
    summary_writer.close()

    print('Total Time:', end_time - start_time)

