
import tensorflow as tf

#Activation Function
def activate(input_layer,act = 'relu',name='activation'):
	if act == None:
		return input_layer
	if act == 'relu':
		return tf.nn.relu(input_layer,name)
	if act =='sqr':
		return tf.square(input_layer,name)
	if act == 'sqr_sigmoid':
		return tf.nn.sigmoid(tf.square(input_layer,name))
	if act=='sigmoid':
		return tf.nn.sigmoid(input_layer,name)


#Fully connected cusom layer
#Supported activation function types : None,relu,sqr,sqr_sigmoid,sigmoid
def fc(input_tensor,n_output_units,scope,activation_fn= 'relu',uniform=False):
	shape = [input_tensor.get_shape().as_list()[-1], n_output_units]
	with tf.variable_scope(scope):
		if uniform:
			weights = tf.Variable(tf.random_uniform(
					shape = shape,
					dtype = tf.float32),
				name='weights')
		else:
			weights = tf.Variable(tf.truncated_normal(
                	shape=shape,
                	mean=0.0,
                	stddev=0.1,
                	dtype=tf.float32),
                name='weights')
		biases = tf.Variable(tf.zeros(shape=[n_output_units]),name='biases', dtype=tf.float32)
		act = tf.matmul(input_tensor, weights) + biases

		return activate(act,activation_fn),weights,biases



