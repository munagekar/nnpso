'''
This file contains the new network definition
Author: Abhishek Munagekar
Language: Python 3
'''
#The number of values for xor


import itertools
from functools import reduce
import operator
import tensorflow as tf
import time
import os

#To force tensorflow-gpu to cpu
#os.environ['CUDA_VISIBLE_DEVICES'] = ''



#Dataset Generation for xor

N_IN = 20
N_BATCHSIZE = 2**N_IN
#A list of lists having N_IN elements all either 0 or 1
xor_in = [list(i) for i in itertools.product([0, 1], repeat=N_IN)]
#A list having 2^N lists each having xor of each input list in the list of lists
xor_out = list(map(lambda x: [(reduce(operator.xor,x))],xor_in))


#Parameters for pso
#TODO: Add batch size and other properties
N_PARTICLES = 32
N_ITERATIONS = int(1e1)


#Basic Neural Network Definition
#Simple feedforward Network

HIDDEN_LAYERS = [10,5,3,2]
HIDDEN_LAYERS.append(1)



net_in = tf.placeholder(dtype=tf.float32,
						shape=[N_BATCHSIZE,N_IN],
						name='net_in')

label = tf.placeholder(dtype=tf.float32,
					   shape=[N_BATCHSIZE,1],
					   name='net_label')

net = net_in
for idx,num_neuron in enumerate(HIDDEN_LAYERS):
	net = tf.contrib.layers.fully_connected(inputs=net,
										   num_outputs=num_neuron,
										   activation_fn =tf.nn.relu,
										   trainable=False,
										   scope = 'fc'+str(idx+1))

#The required output is the net
net_out = net
loss = tf.nn.l2_loss(net_out-label)

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	start_time = time.time()
	for i in range(N_ITERATIONS):
		_loss = sess.run([loss],feed_dict={net_in:xor_in,label:xor_out})

	end_time = time.time()

	print('Total Time:',end_time-start_time)