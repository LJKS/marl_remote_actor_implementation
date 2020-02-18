import models
import ray
import numpy as np
import logging
import tensorflow as tf
logging.getLogger("tensorflow").setLevel(logging.ERROR)
@ray.remote(num_cpus=4)
class PPO_optimizer:
    '''
    @params:
     - data: dict with keys: 'value_targets', 'states', 'actions', 'sampling_action_log_probs', 'advantages'
    '''
    def __init__(self, network_descriptions, actor_weights, critic_weights, data):
        self.epsilon = 0.2
        self.buffer_size = 2000
        self.batch_size = 32
        self.critic_optimization_epochs = 2
        self.actor_optimization_epochs = 5
        self.entropy_coefficient = .01
        actor_description = network_descriptions['actor']
        self.actor = models.Actor(actor_description[0](actor_description[1], actor_description[2], actor_description[3], actor_description[4], actor_description[5]))
        self.actor.set_weights(actor_weights)
        critic_description = network_descriptions['critic']
        self.critic = critic_description[0](critic_description[1], critic_description[2], critic_description[3])
        self.critic.set_weights(critic_weights)
        self.datasets = self.create_datasets(data)

    @ray.method(num_return_vals=2)
    def optimize(self):
        for _ in range(self.actor_optimization_epochs):
            self._optimize_actor_ppo()
        for _ in range(self.critic_optimization_epochs):
            self._optimize_critic()
        return self.actor.model.get_weights(), self.critic.get_weights()

    def _optimize_critic(self):
        #prepare data
        dataset_states = self.datasets['states']
        dataset_value_targets = self.datasets['value_targets']
        dataset_states, dataset_value_targets = self._prepare_datasets(['states', 'value_targets'])

        for states, value_targets in zip(dataset_states, dataset_value_targets):
            with tf.GradientTape() as tape:
                predictions = self.critic(states)
                loss = tf.keras.losses.MSE(value_targets, predictions)
                gradients = tape.gradient(loss, self.critic.trainable_variables)
                self.critic.optimizer.apply_gradients(zip(gradients, self.critic.trainable_variables))

    def _optimize_actor_ppo(self):
        dataset_states, dataset_actions, dataset_sampling_action_log_probs, dataset_advantages = self._prepare_datasets(['states', 'actions', 'sampling_action_log_probs', 'advantages'])
        for states, actions, sampling_action_log_probs, advantages in zip(dataset_states, dataset_actions, dataset_sampling_action_log_probs, dataset_advantages):
            with tf.GradientTape() as tape:
                new_action_log_probs, entropy = self.actor.action_log_prob_and_entropy(states, actions)
                ppo_loss = self._ppo_and_entropy_loss(advantages, sampling_action_log_probs, new_action_log_probs, entropy)
                gradients = tape.gradient(ppo_loss, self.actor.model.trainable_variables)
                self.actor.model.optimizer.apply_gradients(zip(gradients, self.actor.model.trainable_variables))

    def _ppo_and_entropy_loss(self, advantages, sampling_action_log_probs, new_action_log_probs, entropy):
        ppo_objective = self._ppo_objective(advantages, sampling_action_log_probs, new_action_log_probs)
        entropy_objective = self.entropy_coefficient*entropy
        #print('entropy', entropy_objective)
        ppo_and_entropy_loss = - (ppo_objective + entropy_objective)
        return ppo_and_entropy_loss

    def _ppo_objective(self, advantages, sampling_action_log_probs, new_action_log_probs):
        log_probability_ratio = new_action_log_probs - sampling_action_log_probs
        probability_ratio = tf.math.exp(log_probability_ratio)
        #print('advantages', advantages)
        clipped_probability_ratio = tf.clip_by_value(probability_ratio, 1-self.epsilon, 1+self.epsilon)
        ppo_objective = tf.math.minimum(advantages*probability_ratio, advantages*clipped_probability_ratio)
        #print('ppoobjective', ppo_objective)
        return ppo_objective


    def _prepare_datasets(self, keys):
        rand_seed = np.random.randint(20000)
        datasets = []
        for key in keys:
            key_data = self.datasets[key]
            key_data = key_data.shuffle(self.buffer_size, seed=rand_seed)
            key_data = key_data.batch(self.batch_size)
            datasets.append(key_data)
        return datasets

    def create_datasets(self, data):
        datasets = {}
        for key in data.keys():
            if key == 'advantages':
                advs = data[key]
                advs = np.asarray(advs)
                advs = advs - np.mean(advs)
                advs = advs / np.std(advs)
                advs = np.squeeze(advs)
                data[key]=np.split(advs, advs.shape[0])
            datasets[key] = tf.data.Dataset.from_tensor_slices(data[key])
        return datasets
