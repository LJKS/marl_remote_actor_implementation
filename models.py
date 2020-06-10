import logging
import tensorflow as tf
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np


class V_MLP_model(tf.keras.Model):
    """
    V_MLP_model is a class, describing the critic network implemented by a MLP. It's input is the state, the output is the estimation of the state-value - the discounted sum of expected rewards
        @params:
            - network description is a list, where each entry is the num_units of the respective hidden layer and the length is the depth
            - learning_rate is learnrate of model
            - input shape is the expected shape of the network input - used for first initialization
        @call returns estimation of shape [batch_size, 1]
    """
    def __init__(self, network_description, learning_rate, input_size):
        super(V_MLP_model, self).__init__()
        self.network_description = network_description
        self.learning_rate = learning_rate
        self.layer_list = [tf.keras.layers.Dense(num_units, tf.nn.tanh) for num_units in network_description]
        self.value_layer = tf.keras.layers.Dense(1)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        #initialize self
        self(np.zeros(input_size))

    #tf.function optimizes runtime speed
    @tf.function
    def call(self, input):
        x = input
        for i in range(len(self.layer_list)):
            x = self.layer_list[i](x)
        x = self.value_layer(x)
        return x


class MLP_model(tf.keras.Model):
    """
    model is a class, describing the network. It's input is the state, the output is a tfp.distribution over the action space
        @params:
            - network description is a list, where each entry is the num_units of the respective hidden layer and the length is the depth
            - distribution is either 'continuous' or 'discrete' and results in gaussian or respectively categorical distribution
            - out_size is the size of the output, in case of continuous dimensionality of the action, in case of discrete the number of possible actions
            - learning_rate is learnrate of model
        @call returns a tfp Distribution object over action space of shape [batch_size, out_size] of type Categorical if 'discrete', of type Normal if 'continuous'
    """
    def __init__(self, network_description, distribution, out_size, learning_rate, input_size):
        #network_description here is a list containing the number of units of each hidden layer
        super(MLP_model, self).__init__()
        self.network_description = network_description
        self.distribution = distribution
        self.out_size = out_size
        self.learning_rate = learning_rate
        self.layer_list = [tf.keras.layers.Dense(num_units, tf.nn.tanh) for num_units in network_description]
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        if self.distribution == 'continuous':
            self.log_sigma = tf.Variable(-.5)
            self.mean_layer = tf.keras.layers.Dense(out_size, tf.nn.tanh)
        elif self.distribution == 'discrete':
            self.action_logit_layer = tf.keras.layers.Dense(out_size, tf.nn.softmax)
        self(np.zeros(input_size))

    def call(self, x):
        for i in range(len(self.layer_list)):
            x = self.layer_list[i](x)
        if self.distribution == 'continuous':
            x = tfd.Normal(self.mean_layer(x), tf.exp(self.log_sigma), allow_nan_stats=False)
        elif self.distribution == 'discrete':
            x = tf.math.log(self.action_logit_layer(x))
            x = tfd.Categorical(logits=x, allow_nan_stats=False)
        return x

class Dense_CNN_Model(tf.keras.Model):
    def __init__(self, network_description, distribution, out_size, learning_rate, input_size):
        """
        network_description is a dict containing:
            - 'layers' : number of layers in denseblock
            - 'filters' : number of filters per layer in the denseBlock
            - 'mlp' : list containing number of hidden units per layer
        """
        super(Dense_CNN_Model, self).__init__()
        self.network_description = network_description
        self.distribution = distribution
        self.out_size = out_size
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.denseBlock = DenseBlock(layers=self.network_description['layers'], filters=self.network_description['filters'], kernel_size=(3,3), strides=(1,1))
        self.mlp_layers = [tf.keras.layers.Dense(num_units, tf.nn.tanh) for num_units in self.network_description['mlp']]
        if self.distribution == 'continuous':
            self.log_sigma = tf.Variable(-.5)
            self.mean_layer = tf.keras.layers.Dense(out_size, tf.nn.tanh)
        elif self.distribution == 'discrete':
            self.action_logit_layer = tf.keras.layers.Dense(out_size, tf.nn.softmax)
        self(np.zeros(input_size))

    def call(self, x):
        batch_size, _,_,_ = x.shape
        x = self.denseBlock(x)
        x = tf.reshape(x, (batch_size,-1))
        for layer in self.mlp_layers:
            x = layer(x)
        if self.distribution == 'continuous':
            x = tfd.Normal(self.mean_layer(x), tf.exp(self.log_sigma), allow_nan_stats=False)
        elif self.distribution == 'discrete':
            x = tf.math.log(self.action_logit_layer(x))
            x = tfd.Categorical(logits=x, allow_nan_stats=False)
        return x

class V_Dense_CNN_Model(tf.keras.Model):
    def __init__(self, network_description, learning_rate, input_size):
        """
        network_description is a dict containing:
            - 'layers' : number of layers in denseblock
            - 'filters' : number of filters per layer in the denseBlock
            - 'mlp' : list containing number of hidden units per layer
        """
        super(V_Dense_CNN_Model, self).__init__()
        self.network_description = network_description
        self.learning_rate = learning_rate
        self.input_size = input_size
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.denseBlock = DenseBlock(layers=self.network_description['layers'], filters=self.network_description['filters'], kernel_size=(3,3), strides=(1,1))
        self.mlp_layers = [tf.keras.layers.Dense(num_units, tf.nn.tanh) for num_units in self.network_description['mlp']]
        self.value_layer = tf.keras.layers.Dense(1)
        self(np.zeros(input_size))

    def call(self, x):
        batch_size, _,_,_ = x.shape
        x = self.denseBlock(x)
        x = tf.reshape(x, (batch_size,-1))
        for layer in self.mlp_layers:
            x = layer(x)
        x = self.value_layer(x)
        return x


class DenseBlock(tf.keras.layers.Layer):
    def __init__(self, layers, filters, kernel_size, strides, bottleneck=False):
        super(DenseBlock, self).__init__()
        self.has_bottleneck = bottleneck!=False

        if bottleneck!=False:
            self.layers = [BNDenseLayer(filters, kernel_size, strides, bottleneck) for _ in range(layers)]
        else:
            self.layers = [DenseLayer(filters, kernel_size, strides) for _ in range(layers)]

    def call(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DenseLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides):
        super(DenseLayer, self).__init__()
        self.convolutions = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding='SAME', activation=tf.nn.relu)

    def call(self, x):
        new_feature_maps = self.convolutions(x)
        feature_maps = tf.concat([new_feature_maps, x], -1)
        return feature_maps


class BNDenseLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, bottleneck_size):
        super(BNDenseLayer, self).__init__()
        self.bottleneck = tf.keras.layers.Conv2D(bottleneck_size, (1,1), (1,1), padding='SAME', activation=tf.nn.relu)
        self.convolutions = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding='SAME', activation=tf.nn.relu)

    def call(self, x):
        bottlenecked = self.bottleneck(x)
        new_feature_maps = self.convolutions(bottlenecked)
        feature_maps = tf.concat([new_feature_maps, x], -1)
        return feature_maps

class Actor:
    """
    wrapper for model class to make calls to sampling, log probs and entropies possible
    """
    def __init__(self, model):
        self.model = model


    @tf.function
    def act(self, state):
        """
        function designed for sampling actions while gathering respective log probalities
        @params:
            - state: state of shape [batch_size, input_size]
        @returns:
            - action sampled from model output distribution,
            - respective log probability
        """
        action_distribution = self.model(state)
        action = action_distribution.sample()
        log_prob = action_distribution.log_prob(action)
        if self.model.distribution == 'continuous':
            log_prob = tf.reduce_sum(log_prob, axis=-1)
        return action, log_prob


    @tf.function
    def action_log_prob_and_entropy(self, state, action):
        """
        function designed for optimizing network policy gradient style
        @params:
            - state: state of shape [batch_size, input_size]
            - action: action as sampled before by the network in the respective state
        @returns:
            - log prob of sampling given action in given state from the distribution described by model
            - respective entropy of the distribution
        """
        action_distribution = self.model(state)
        log_prob = action_distribution.log_prob(action)
        entropy = action_distribution.entropy()

        #if continuous, action distribution is multidimensional - total probability is the product of single probabilities, therefore total log prob is the sum of single log probs
        if self.model.distribution == 'continuous':
            log_prob = tf.reduce_sum(log_prob, axis=-1, keepdims=True)
            entropy = tf.reduce_mean(entropy, axis=-1, keepdims=True)
        elif self.model.distribution == 'discrete':
            log_prob = tf.reshape(log_prob, (-1,1))
            entropy = tf.reshape(entropy, (-1,1))
        return log_prob, entropy

    def load(self, path):
        self.model.load_weights(path)

    def save(self, path):
        self.model.save_weights(path, save_format='tf')

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()
