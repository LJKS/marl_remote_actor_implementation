import model_factory
import ray
import numpy as np
from scipy import signal

@ray.remote(num_cpus=1)
class Remote_Runner:
    """
    @params:
        - runner_id is the id of the remote actor
        - gym_func is a function that returns an instance of the multi agent gym to be used
        - network_description is a dict with key-value pairs:
            'actor'    : [model_class, network_description, distribution, outsize, learning_rate, input_shape]
            'critic'   : [model_class, network_description, learning_rate, input_shape]
            'opponent' : [model_class, network_description, learning_rate, input_shape]
        - actor_weights is is the list of weights for the actor as returned by model.get_weights()
        - critic_weights is is the list of weights for the actor as returned by model.get_weights()
        - opponent_weights is is the list of weights for the actor as returned by model.get_weights()
    """
    def __init__(self, runner_id, opponent_id, gym_class, network_descriptions, actor_weights, critic_weights, opponent_weights, gam, lam):
        self.runner_id = runner_id
        self.opponent_id = opponent_id
        self.gym_class = gym_class
        actor_description = network_descriptions['actor']
        self.actor = model_factory.get_model('Actor')(model_factory.get_model(actor_description[0])(actor_description[1], actor_description[2], actor_description[3], actor_description[4], actor_description[5]))
        self.actor.set_weights(actor_weights)
        critic_description = network_descriptions['critic']
        self.critic = model_factory.get_model(critic_description[0])(critic_description[1], critic_description[2], critic_description[3])
        self.critic.set_weights(critic_weights)
        opponent_description = network_descriptions['opponent']
        self.opponent = model_factory.get_model('Actor')(model_factory.get_model(opponent_description[0])(opponent_description[1], opponent_description[2], opponent_description[3], opponent_description[4], opponent_description[5]))
        self.opponent.set_weights(opponent_weights)
        self.gym = gym_class(self.opponent)
        self.gam = gam
        self.lam = lam

    @ray.method(num_return_vals=1)
    def generate_sample(self):
        states, actions, action_log_probs, value_targets, advantages, rewards = self.run()
        agent_won = self.gym.agent_won()
        #Just return one object_id containing all elements - makes using ray.wait() more intuitive
        return (states, actions, action_log_probs, value_targets, advantages, rewards, agent_won, self.runner_id, self.opponent_id)

    def run(self):
        states = []
        rewards = []
        actions = []
        action_log_probs = []
        value_estimates = []
        done = False
        state = self.gym.reset()
        states.append(state)
        while not done:
            action, log_prob = self.actor.act(state)
            action = action.numpy()
            state, reward, done, _ = self.gym.step(action)
            value_estimate = self.critic(state)
            if not done:
                states.append(state)
            rewards.append(reward)
            actions.append(action)
            action_log_probs.append(log_prob)
            value_estimates.append(np.reshape(value_estimate.numpy(),()))
        rewards = np.asarray(rewards)
        value_estimates = np.asarray(value_estimates)
        value_targets = np.reshape(self.compute_value_targets(rewards),(-1,1))
        advantages = np.reshape(self.compute_gaes(rewards, value_estimate), (-1,1))
        states = np.concatenate(states)
        actions = np.concatenate(actions)
        action_log_probs = np.reshape(np.concatenate(action_log_probs), (-1,1))
        return states, actions, action_log_probs, value_targets, advantages, rewards

    def compute_value_targets(self, rewards):
        value_targets = self.discount_cumsum(rewards, self.gam)
        return value_targets

    def compute_gaes(self, rewards, values):
        values_app_zero = np.append(values, 0)
        td_errors = rewards + self.gam*values_app_zero[1:] - values_app_zero[:-1]
        gam_lam = self.gam * self.lam
        gae_advantages = self.discount_cumsum(td_errors, gam_lam)
        return gae_advantages.astype(np.float32)

    def discount_cumsum(self, x, discount):
        """
        magic from rllab for computing discounted cumulative sums of vectors.
        input:
            vector x,
            [x0,
             x1,
             x2]
        output:
            [x0 + discount * x1 + discount^2 * x2,
             x1 + discount * x2,
             x2]
        """
        return signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
