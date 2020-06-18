import ray
import numpy as np
import logging
import model_factory
import tensorflow as tf
logging.getLogger("tensorflow").setLevel(logging.ERROR)

class PPO_optimizer:
    '''
    @params:
     - network_descriptions: descripions of the actor and critic model
     - actor weights: weights for the actor ANN
     - critic weights: weights for the critic ANN
     - data: dict with keys: 'value_targets', 'states', 'actions', 'sampling_action_log_probs', 'advantages'
    '''

    def __init__(self, network_descriptions, actor_weights, critic_weights, data, hyperparameters):
        self.data_keys = ['value_targets', 'states', 'actions', 'sampling_action_log_probs', 'advantages']
        self.epsilon = 0.2
        self.shuffle_buffer_size = 2000
        self.prefetch_buffer_size = 2000
        self.batch_size = 64
        self.critic_optimization_epochs = 20
        self.actor_optimization_epochs = 20
        self.entropy_coefficient = 0.01
        actor_description = network_descriptions['actor']
        self.actor = model_factory.get_model('Actor')(model_factory.get_model(actor_description[0])(actor_description[1], actor_description[2], actor_description[3], actor_description[4], actor_description[5]))
        self.actor.set_weights(actor_weights)
        critic_description = network_descriptions['critic']
        self.critic = model_factory.get_model(critic_description[0])(critic_description[1], critic_description[2], critic_description[3])
        self.critic.set_weights(critic_weights)
        self.dataset = self.create_datasets(data)
        #debug
        self.critic_loss_logger=[]
        self.entropy_logger=[]
        self.ppo_policy_loss_logger=[]



    def optimize(self):
        for _ in range(self.actor_optimization_epochs):
            self._optimize_actor_ppo()
        for _ in range(self.critic_optimization_epochs):
            self._optimize_critic()
        report = {'critic_history':self.critic_loss_logger, 'critic_summary': np.mean(np.asarray(self.critic_loss_logger)), 'entropy_history':[entropy for entropy in self.entropy_logger], 'entropy_summary': np.mean(np.asarray(self.entropy_logger)), 'policy_history': self.ppo_policy_loss_logger, 'policy_summary': np.mean(np.asarray(self.ppo_policy_loss_logger))}
        #report = {'critic_history':self.critic_loss_logger, 'critic_summary': np.mean(np.asarray(self.critic_loss_logger)), 'entropy_history':[entropy/self.entropy_coefficient for entropy in self.entropy_logger], 'entropy_summary': np.mean(np.asarray(self.entropy_logger))/self.entropy_coefficient, 'policy_history': self.ppo_policy_loss_logger, 'policy_summary': np.mean(np.asarray(self.ppo_policy_loss_logger))}
        return self.actor.model.get_weights(), self.critic.get_weights(), report

    def _optimize_critic(self):
        losses = []
        for value_targets, states, _, _, _  in self.dataset:
            loss = self._optimize_critic_step(value_targets, states)
            losses.append(np.mean(loss.numpy()))
        self.critic_loss_logger.append(np.mean(np.asarray(losses)))

    @tf.function
    def _optimize_critic_step(self, value_targets, states):

        with tf.GradientTape() as tape:
            predictions = self.critic(states)
            #print('state shape', states.numpy().shape)
            #print('pred shapes', predictions.numpy().shape)
            #print('target_shapes', value_targets.shape)
            loss = tf.keras.losses.MSE(value_targets, predictions)
            loss = tf.reduce_mean(loss)
            #print('loss shape', loss.numpy().shape)
            gradients = tape.gradient(loss, self.critic.trainable_variables)
            self.critic.optimizer.apply_gradients(zip(gradients, self.critic.trainable_variables))
            return loss

    def _optimize_actor_ppo(self):
        entropies = []
        policy_losses = []
        for _, states, actions, sampling_action_log_probs, advantages in self.dataset:
            #print('stateshape', states.shape)
            #print('action_shape', actions.shape)
            #print('sampling_action_log_probs_shape', sampling_action_log_probs.shape)
            #print('adv_shape', advantages.shape)
            policy_loss, entropy_loss = self._optimize_actor_ppo_step(states, actions, sampling_action_log_probs, advantages)
            entropies.append(np.mean(entropy_loss.numpy()))
            policy_losses.append(np.mean(policy_loss.numpy()))
        self.entropy_logger.append(-np.mean(np.asarray(entropies)))
        #self.ppo_policy_loss_logger.append(-np.mean(np.asarray(policy_losses)))
        self.ppo_policy_loss_logger.append(np.mean(np.abs(np.asarray(policy_losses))))


    @tf.function
    def _optimize_actor_ppo_step(self, states, actions, sampling_action_log_probs, advantages):
        with tf.GradientTape() as tape:
            new_action_log_probs, entropy = self.actor.action_log_prob_and_entropy(states, actions)
            #print('new action log probs', new_action_log_probs.numpy().shape)
            #print('entropy shapoe', entropy.numpy().shape)
            ppo_loss, entropy_loss = self._ppo_and_entropy_loss(advantages, sampling_action_log_probs, new_action_log_probs, entropy)
            sum_ppo_policy_loss = tf.reduce_mean(ppo_loss) + tf.reduce_mean(entropy_loss)
            #print('ppo_loss_shape', ppo_loss.numpy().shape)
            gradients = tape.gradient(sum_ppo_policy_loss, self.actor.model.trainable_variables)
            self.actor.model.optimizer.apply_gradients(zip(gradients, self.actor.model.trainable_variables))
            return ppo_loss, entropy_loss

    def _ppo_and_entropy_loss(self, advantages, sampling_action_log_probs, new_action_log_probs, entropy):
        #print('advantages shape', advantages.numpy().shape)
        #print('sampling_action_log_probs shape', sampling_action_log_probs.numpy().shape)
        ppo_objective = - self._ppo_objective(advantages, sampling_action_log_probs, new_action_log_probs)
        #print('ppo objective shape', ppo_objective.numpy().shape)
        entropy_objective = - self.entropy_coefficient*entropy
        #print('entropy_objective_shape', entropy_objective.numpy().shape)
        return ppo_objective, entropy_objective

    def _ppo_objective(self, advantages, sampling_action_log_probs, new_action_log_probs):
        log_probability_ratio = new_action_log_probs - sampling_action_log_probs
        #print('log_prob_ratio_shape', log_probability_ratio.numpy().shape)
        probability_ratio = tf.math.exp(log_probability_ratio)
        #print('probration_shape', probability_ratio.numpy().shape)
        clipped_probability_ratio = tf.clip_by_value(probability_ratio, 1-self.epsilon, 1+self.epsilon)
        #print('clipped_probability_ratio_shape', clipped_probability_ratio.numpy().shape)
        ppo_objective = tf.math.minimum(advantages*probability_ratio, advantages*clipped_probability_ratio)
        #print('_ppo_objective_shape', ppo_objective.numpy().shape)
        return ppo_objective

    def create_datasets(self, data):
        data['advantages'] = data['advantages'] - np.mean(data['advantages'])
        data['advantages'] = data['advantages'] / np.std(data['advantages'])
        dataset = tf.data.Dataset.from_tensor_slices(tuple([data[key] for key in self.data_keys]))
        dataset = dataset.shuffle(self.shuffle_buffer_size)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(self.prefetch_buffer_size)
        return dataset
