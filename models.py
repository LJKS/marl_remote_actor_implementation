import logging
import tensorflow as tf
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np

MIN_SIGMA = 0.2

class V_MLP_model(tf.keras.Model):
    def __init__(self, network_description, learning_rate, input_shape):
        super(V_MLP_model, self).__init__()
        self.network_description = network_description
        self.learning_rate = learning_rate
        self.layer_list = [tf.keras.layers.Dense(num_units, tf.nn.tanh) for num_units in network_description]
        self.value_layer = tf.keras.layers.Dense(1)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self(np.zeros(input_shape))

    @tf.function
    def call(self, input):
        x = input
        for i in range(len(self.layer_list)):
            x = self.layer_list[i](x)
        x = self.value_layer(x)
        return x

"""
@params:
    model is a class, describing the network. It's input is the state, the output is a tfp.distribution over the action space
    @params:
        - network description is a list, where each entry is the num_units of the respective hidden layer and the length is the depth
        - distribution is either 'continuous' or 'discrete' and results in gaussian or respectively categorical distribution
        - out_size is the size of the output, in case of continuous dimensionality of the action, in case of discrete the number of possible actions
        - learning_rate is learnrate of model
    @call returns Distribution over action space of shape [batch_size, outsize] if 'continuous', of shape [batch_size, 1] else
"""
class MLP_model(tf.keras.Model):
    def __init__(self, network_description, distribution, out_size, learning_rate, input_shape):
        super(MLP_model, self).__init__()
        self.network_description = network_description
        self.distribution = distribution
        self.out_size = out_size
        self.learning_rate = learning_rate
        self.layer_list = [tf.keras.layers.Dense(num_units, tf.nn.tanh) for num_units in network_description]
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        if self.distribution == 'continuous':
            self.mean_layer = tf.keras.layers.Dense(out_size, tf.nn.tanh)
            self.log_sigma_layer = tf.keras.layers.Dense(out_size)
        elif self.distribution == 'discrete':
            self.action_logit_layer = tf.keras.layers.Dense(out_size, tf.nn.softmax)
        self(np.zeros(input_shape))

    def call(self, x):
        for i in range(len(self.layer_list)):
            x = self.layer_list[i](x)
        if self.distribution == 'continuous':
            x = tfd.Normal(self.mean_layer(x), tf.exp(self.log_sigma_layer(x)), allow_nan_stats=False)
        elif self.distribution == 'discrete':
            x = tf.math.log(self.action_logit_layer(x))
            x = tfd.Categorical(logits=x, allow_nan_stats=False)
        return x

class Actor:
    def __init__(self, model):
        self.model = model

    @tf.function
    def act(self, state):
        action_distribution = self.model(state)
        action = action_distribution.sample()
        log_prob = action_distribution.log_prob(action)
        if self.model.distribution == 'continuous':
            log_prob = tf.reduce_sum(log_prob, axis=-1)
        return action, log_prob

    #returns log probability of action given state
    @tf.function
    def action_log_prob_and_entropy(self, state, action):
        action_distribution = self.model(state)
        log_prob = action_distribution.log_prob(action)
        entropy = action_distribution.entropy()

        if self.model.distribution == 'continuous':
            log_prob = tf.reduce_sum(log_prob, axis=-1, keepdims=True)
            entropy = tf.reduce_sum(entropy, axis=-1, keepdims=True)
        return log_prob, entropy

    #load&&save: use tf save type, as it also saves the optimizer
    def load(self, path):
        self.model.load_weights(path)

    def save(self, path):
        self.model.save_weights(path, save_format='tf')

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()
