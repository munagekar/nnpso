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
import layers


#Dataset Generation for xor

N_IN = 5
N_BATCHSIZE = 32


#Parameters for pso
#TODO: Add batch size and other properties
N_PARTICLES = 4
N_ITERATIONS = int(1e5)
LEARNING_RATE = 0.001


#Basic Neural Network Definition
#Simple feedforward Network

HIDDEN_LAYERS = [3,2]
LAYERS = [N_IN]+HIDDEN_LAYERS+[1]
print ('The Network Structure is',LAYERS)



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

print ('Input & Label Set')
print ('Starting to Build Network')

###MULTI-PARTICLE NEURAL NETS

losses = []
nets = []
train_ops =[]


for pno in range(N_PARTICLES):
	net = net_in

	for idx,num_neuron in enumerate(HIDDEN_LAYERS):
		layer_scope = 'pno'+str(pno+1)+'fc'+str(idx+1)
		net,w,b = layers.fc(input_tensor=net,
						n_output_units =num_neuron,
						activation_fn='sigmoid',
						scope = layer_scope,
						uniform = False)

	#Define loss for each of the particle nets
	loss = tf.nn.l2_loss(net - label)
	#Define an optimizer for the particle nets
	optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
	train_op = optimizer.minimize(loss)
	nets.append(net)
	losses.append(loss)
	train_ops.append(train_op)



print('Network Build Successful')


#Initialize the entire graph
init = tf.global_variables_initializer()
print ('Graph Init Successful')


#for var in tf.global_variables():
#	print (var)



req_list = losses + train_ops



with tf.Session() as sess:
	sess.run(init)
	start_time = time.time()
	for i in range(N_ITERATIONS):
		#xor_in,xor_out = xor_next_batch(N_BATCHSIZE,N_IN)
		dict_out= sess.run(req_list,feed_dict={net_in:xor_in,label:xor_out})
		
		if i%10000 == 0:
			_losses = dict_out[:len(dict_out)//2]
			print ('Losses:',_losses,'Iteration:',i)

	end_time = time.time()

	print('Total Time:',end_time-start_time)
