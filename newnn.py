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
N_PARTICLES = 4
N_ITERATIONS = int(1e5)
P_BEST_FACTOR = 2
G_BEST_FACTOR = 2
# Velocity Decay specifies the multiplier for the velocity update
VELOCITY_DECAY = 1.0


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

print('Input & Label Set')
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

# Fixed Constant


# TODO:Parellized the following loop
for pno in range(N_PARTICLES):
    net = net_in

    for idx, num_neuron in enumerate(LAYERS[1:]):
        layer_scope = 'pno' + str(pno + 1) + 'fc' + str(idx + 1)
        net, w, b, pw, pb, pf, vw, vb = layers.fc(input_tensor=net,
                                                  n_output_units=num_neuron,
                                                  activation_fn='sigmoid',
                                                  scope=layer_scope,
                                                  uniform=True)

        # Constants & Other Random Variables
        pbestrand = tf.random_uniform(shape=[],
                                      maxval=P_BEST_FACTOR,
                                      name=layer_scope + 'pbestrand')

        gbestrand = tf.random_uniform(shape=[],
                                      maxval=G_BEST_FACTOR,
                                      name=layer_scope + 'gbestrand')

        # Multiply by the Velocity Decay
        nextvw = tf.multiply(vw, VELOCITY_DECAY)
        nexvb = tf.multiply(vb, VELOCITY_DECAY)
        # Differences between Particle Best & Current
        pdiffw = tf.subtract(pw, w)
        pdiffb = tf.subtract(pb, b)
        # Define & Reuse the GBest
        with tf.variable_scope("gbest", reuse=tf.AUTO_REUSE):
            gw = tf.get_variable(name='fc' + str(idx + 1) + 'w',
                                 shape=[LAYERS[idx], LAYERS[idx + 1]],
                                 initializer=tf.zeros_initializer)

            gb = tf.get_variable(name='fc' + str(idx + 1) + 'b',
                                 shape=[LAYERS[idx + 1]],
                                 initializer=tf.zeros_initializer)

    # Define loss for each of the particle nets
    loss = tf.nn.l2_loss(net - label)
    # Update the lists
    nets.append(net)
    losses.append(loss)
    print('Building Net:', pno + 1)


print('Network Build Successful')


# Initialize the entire graph
init = tf.global_variables_initializer()
print('Graph Init Successful')

for var in tf.global_variables():
    print(var)


req_list = losses


with tf.Session() as sess:
    sess.run(init)
    start_time = time.time()
    for i in range(N_ITERATIONS):
        #xor_in,xor_out = xor_next_batch(N_BATCHSIZE,N_IN)
        dict_out = sess.run(req_list, feed_dict={
                            net_in: xor_in, label: xor_out})

        if i % 10000 == 0:
            _losses = dict_out
            print('Losses:', _losses, 'Iteration:', i)

    end_time = time.time()

    print('Total Time:', end_time - start_time)
