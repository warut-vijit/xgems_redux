import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl

class estimator(object):
    def __init__(self,reg_penalty=0.):
        self.x_dim = 20
        self.n_labels = 2
        self._name = None
        self.reg_penalty=reg_penalty

    def __call__(self,xin,target=None,reuse=False):
        with tf.variable_scope(self._name) as vs:
            if reuse:
                vs.reuse_variables()
            self.W = tf.get_variable(name="W",shape=[self.x_dim,self.n_labels])
            self.b = tf.get_variable(name="b",shape=[self.n_labels])
            self.model_output = tf.matmul(xin,self.W)+self.b
            self.y=tf.nn.softmax(self.model_output)
            self.target = target
            self.crossentropy=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.model_output,labels=self.target))
            self.l1_penalty = tf.contrib.layers.apply_regularization(tf.contrib.layers.l1_regularizer(scale=self.reg_penalty),[self.W])
            self.crossentropy = self.crossentropy + self.l1_penalty
            return self.y

    def pred(self,xin,reuse=False):
        with tf.variable_scope(self._name) as vs:
            if reuse:
                vs.reuse_variables()
            self.model_output = tf.matmul(xin,self.W)+self.b
            self.y=tf.nn.softmax(self.model_output)
            return self.y

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self,val_str):
        self._name = val_str

    def train(self,learning_rate):
        return tf.train.GradientDescentOptimizer(learning_rate).minimize(self.crossentropy)

    def set_params(self,params):
        self.params=params

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self._name in var.name]

    def loss(self):
        return tf.reduce_mean(-tf.reduce_sum(self.target*tf.log(self.y),reduction_indices=[1]))

