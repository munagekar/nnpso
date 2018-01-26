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
import math


# Dataset Generation for xor

N_IN = 5
N_BATCHSIZE = 32


# Parameters for pso
# TODO: Add batch size and other properties
N_PARTICLES = 64
N_ITERATIONS = int(1000)
P_BEST_FACTOR = 0.05
G_BEST_FACTOR = 0.05
# Velocity Decay specifies the multiplier for the velocity update
VELOCITY_DECAY = 0.999
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
pweights = []
pbiases = []
vweights = []
vbiases = []

random_values = []

# Positional Updates
bias_updates = []
weight_updates = []

# Velocity Updates
vweight_updates = []
vbias_updates = []

# Fitness Updates
fit_updates = []

# Global Best
gweights = []
gbiases = []
gfit = tf.Variable(math.inf, name='gbestfit')

# TODO:Parellized the following loop
# TODO:See if the Conditional Function Lambdas can be optimized
for pno in range(N_PARTICLES):
    weights = []
    biases = []
    pweights = []
    pbiases = []
    pbestrand = tf.Variable(tf.random_uniform(
        shape=[], maxval=P_BEST_FACTOR), name='pno' + str(pno + 1) + 'pbestrand')
    gbestrand = tf.Variable(tf.random_uniform(
        shape=[], maxval=G_BEST_FACTOR), name='pno' + str(pno + 1) + 'gbestrand')
    # Append the random values so that the initializer can be called again
    random_values.append(pbestrand)
    random_values.append(gbestrand)
    pfit = tf.Variable(math.inf, name='pno' + str(pno + 1) + 'fit')

    net = net_in
    # Define the parameters

    for idx, num_neuron in enumerate(LAYERS[1:]):
        layer_scope = 'pno' + str(pno + 1) + 'fc' + str(idx + 1)
        net, w, b, pw, pb, vw, vb = layers.fc(input_tensor=net,
                                              n_output_units=num_neuron,
                                              activation_fn='sigmoid',
                                              scope=layer_scope,
                                              uniform=False)
        vweights.append(vw)
        vbiases.append(vb)
        weights.append(w)
        biases.append(b)
        pweights.append(pw)
        pbiases.append(pb)

        # Multiply by the Velocity Decay
        nextvw = tf.multiply(vw, t_VELOCITY_DECAY)
        nextvb = tf.multiply(vb, t_VELOCITY_DECAY)

        # Differences between Particle Best & Current
        pdiffw = tf.multiply(tf.subtract(pw, w), pbestrand)
        pdiffb = tf.multiply(tf.subtract(pb, b), pbestrand)
        # Define & Reuse the GBest
        gw = None
        gb = None
        with tf.variable_scope("gbest", reuse=tf.AUTO_REUSE):
            gw = tf.get_variable(name='fc' + str(idx + 1) + 'w',
                                 shape=[LAYERS[idx], LAYERS[idx + 1]],
                                 initializer=tf.zeros_initializer)

            gb = tf.get_variable(name='fc' + str(idx + 1) + 'b',
                                 shape=[LAYERS[idx + 1]],
                                 initializer=tf.zeros_initializer)

        # If first Particle add to Global Else it is already present
        if pno == 0:
            gweights.append(gw)
            gbiases.append(gb)

        # Differences between Global Best & Current
        gdiffw = tf.multiply(tf.subtract(gw, w), gbestrand)
        gdiffb = tf.multiply(tf.subtract(gb, b), gbestrand)

        vweight_update = tf.assign(vw,
                                   tf.add_n([nextvw, pdiffw, gdiffw]),
                                   validate_shape=True)
        vweight_updates.append(vweight_update)
        vbias_update = tf.assign(vb,
                                 tf.add_n([nextvb, pdiffb, gdiffb]),
                                 validate_shape=True)
        vbias_updates.append(vbias_update)
        weight_update = tf.assign(w, w + vw, validate_shape=True)
        weight_updates.append(weight_update)
        bias_update = tf.assign(b, b + vb, validate_shape=True)
        bias_updates.append(bias_update)

    # Define loss for each of the particle nets
    loss = tf.nn.l2_loss(net - label)
    particlebest = tf.cond(loss < pfit, lambda: loss, lambda: pfit)
    fit_update = tf.assign(pfit, particlebest, validate_shape=True)
    fit_updates.append(fit_update)
    globalbest = tf.cond(loss < gfit, lambda: loss, lambda: gfit)
    fit_update = tf.assign(gfit, globalbest, validate_shape=True)
    fit_updates.append(fit_update)

    # Multiple Length Checks
    assert len(weights) == len(biases)
    assert len(gweights) == len(gbiases)
    assert len(pweights) == len(pbiases)
    assert len(gweights) == len(weights)
    assert len(pweights) == len(weights)

    for i in range(len(weights)):
        #Particle Best
        pweight = tf.cond(loss < pfit, lambda: weights[i], lambda: pweights[i])
        fit_update = tf.assign(pweights[i], pweight, validate_shape=True)
        fit_updates.append(fit_update)
        pbias = tf.cond(loss<pfit, lambda:biases[i],lambda:pbiases[i])
        fit_update = tf.assign(pbiases[i],pbias,validate_shape=True)
        fit_updates.append(fit_update)

        #Global Best
        gweight = tf.cond(loss < gfit, lambda: weights[i], lambda: gweights[i])
        fit_update = tf.assign(gweights[i], gweight, validate_shape=True)
        fit_updates.append(fit_update)
        gbias = tf.cond(loss<gfit, lambda:biases[i],lambda:gbiases[i])
        fit_update = tf.assign(gbiases[i],gbias,validate_shape=True)
        fit_updates.append(fit_update)


    # Update the lists
    nets.append(net)
    losses.append(loss)
    print('Building Net:', pno + 1)


print('Network Build Successful')
print("Number of Random Values:", len(random_values))
print("Number of Fitness Updates:", len(fit_updates))


# Initialize the entire graph
init = tf.global_variables_initializer()
print('Graph Init Successful:')

for var in tf.global_variables():
    print(var)


# Define the updates which are to be done before each iterations
random_updates = [r.initializer for r in random_values]
updates = weight_updates + bias_updates + \
    random_updates + vbias_updates + vweight_updates + fit_updates
req_list = losses, updates,gfit,gbiases

with tf.Session() as sess:
    sess.run(init)

    # Write The graph summary
    summary_writer = tf.summary.FileWriter('/tmp/tf/logs', sess.graph_def)
    start_time = time.time()
    for i in range(N_ITERATIONS):
        # Reinitialize the Random Values at each iteration

        # xor_in,xor_out = xor_next_batch(N_BATCHSIZE,N_IN)
        dict_out, _ ,gfit,gbiases= sess.run(req_list, feed_dict={
            net_in: xor_in, label: xor_out})

        _losses = dict_out
        print('Losses:', _losses, 'Iteration:', i)
        print('Gfit:',gfit,':',gbiases)

    end_time = time.time()
    # Close the writer
    summary_writer.close()

    print('Total Time:', end_time - start_time)
