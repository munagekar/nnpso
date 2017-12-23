import tensorflow as tf
import random
#Parameters 
learning_rate = 0.001
training_epochs = 10000
batch_size = 1
display_step =1
_seed = 3

#Network Parameters
n_hidden_1 = 5
n_hidden_2 = 5
n_hidden_3 = 5
n_hidden_4 = 5
n_hidden_5 = 5
n_input = 5 #Number of numbers for xor
n_classes =1 # Answer will either be 0 or 1 for binary one class is sufficient



#xor stuff

def xor_init():
	random.seed(_seed)
def xor_next_batch(batch_size):
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

	





#tf Graph Input
X = tf.placeholder(tf.float32,shape = [None,n_input])
Y = tf.placeholder(tf.float32,shape = [None,n_classes])

weights = {
	'h1': tf.Variable(tf.random_normal([n_input,n_hidden_1])),
	'h2': tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
	'h3': tf.Variable(tf.random_normal([n_hidden_2,n_hidden_3])),
	'h4': tf.Variable(tf.random_normal([n_hidden_3,n_hidden_4])),
	'h5': tf.Variable(tf.random_normal([n_hidden_4,n_hidden_5])),
	'out': tf.Variable(tf.random_normal([n_hidden_5,n_classes]))
}

biases = {
	
	'b1' : tf.Variable(tf.random_normal([n_hidden_1])),
	'b2' : tf.Variable(tf.random_normal([n_hidden_2])),
	'b3' : tf.Variable(tf.random_normal([n_hidden_3])),
	'b4' : tf.Variable(tf.random_normal([n_hidden_4])),
	'b5' : tf.Variable(tf.random_normal([n_hidden_5])),
	'out' : tf.Variable(tf.random_normal([n_classes]))
}

def multilayer_perceptron(x):
	layer_1 = tf.add(tf.matmul(x,weights['h1']),biases['b1'])
	layer_2 = tf.add(tf.matmul(layer_1,weights['h2']),biases['b2'])
	layer_3 = tf.add(tf.matmul(layer_2,weights['h3']),biases['b3'])
	layer_4 = tf.add(tf.matmul(layer_3,weights['h4']),biases['b4'])
	layer_5 = tf.add(tf.matmul(layer_4,weights['h5']),biases['b5'])
	out_layer = tf.add(tf.matmul(layer_5,weights['out']),biases['out'])
	return tf.sigmoid(out_layer) #Sigmoid Activation for scaling zero to one

#Construction of the model
logits =multilayer_perceptron(X)

#Define the losses 
loss_op = tf.losses.mean_squared_error(labels=Y,predictions=logits)

#Optimization
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

#Initialize
init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)


	for epoch in range(training_epochs):
		avg_cost = 0
		total_batch = int(10000/batch_size)
		for i in range(total_batch):
			batch_x,batch_y = xor_next_batch(batch_size)
			_,c =sess.run([train_op,loss_op],feed_dict={X:batch_x,Y:batch_y})
			avg_cost +=c/total_batch
		if epoch % display_step ==0:
			print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))

