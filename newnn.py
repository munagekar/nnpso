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
import random


#Dataset Generation for xor

N_IN = 5
N_BATCHSIZE = 32


#Parameters for pso
#TODO: Add batch size and other properties
N_PARTICLES = 32
N_ITERATIONS = int(100)
LEARNING_RATE = 0.001


#Basic Neural Network Definition
#Simple feedforward Network

HIDDEN_LAYERS = [3,2]
HIDDEN_LAYERS.append(1)

#Utility Functions

#Activation Function
def activate(input_layer,act = 'relu',name='activation'):
	if act == None:
		return input_layer
	if act == 'relu':
		return tf.nn.relu(input_layer,name)
	if act =='sqr':
		return tf.square(input_layer,name)
	if act == 'sqr_relu':
		return tf.nn.relu(tf.square(input_layer,name))
	if act == 'sqr_sigmoid':
		return tf.nn.sigmoid(tf.square(input_layer,name))
	if act=='sigmoid':
		return tf.nn.sigmoid(input_layer,name)

#Xorgenerator Function
def xor_next_batch(batch_size,n_input):
	batch_x = []
	batch_y = []
	for i in range(batch_size):
		x=[]
		y=[]
		ans = 0
		for j in range (n_input):
			x.append(random.randint(0,1))
			ans^=x[j]
		y.append(ans)
		batch_y.append(y)
		batch_x.append(x)
	return batch_x,batch_y


#A list of lists having N_IN elements all either 0 or 1
xor_in = [list(i) for i in itertools.product([0, 1], repeat=N_IN)]
#A list having 2^N lists each having xor of each input list in the list of lists
xor_out = list(map(lambda x: [(reduce(operator.xor,x))],xor_in))


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
										   activation_fn =None,
										   trainable=True,
										   scope = 'fc'+str(idx+1)
										   )
	net = activate(net,'sigmoid',name='act_'+str(idx+1))

#The required output is the net
net_out = net

#Define the losses
loss = tf.nn.l2_loss(net_out-label)

#Define the optim step
optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train_op = optimizer.minimize(loss)


init = tf.global_variables_initializer()


for var in tf.global_variables():
	print (var)



with tf.Session() as sess:
	sess.run(init)
	start_time = time.time()
	for i in range(N_ITERATIONS):
		#xor_in,xor_out = xor_next_batch(N_BATCHSIZE,N_IN)
		_,_loss = sess.run([train_op,loss],feed_dict={net_in:xor_in,label:xor_out})
		
		with tf.variable_scope("fc1", reuse=True):
			myvar =tf.get_variable("weights")
			myvar = tf.get_variable("biases")
			print(myvar)
		if i%1000 == 0:
			print ('Iteration:',i,'Loss',_loss)

	end_time = time.time()

	print('Total Time:',end_time-start_time)

