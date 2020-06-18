import ppo
import ray
"""
binding to ppo optimizer class
"""
def optimize_ppo(network_descriptions, actor_weights, critic_weights, data, hyperparameters):
    # Optimizes Actor and Critic Network using PPO on the number of available cpus and gpus as given by hyperparameters
    @ray.remote(num_cpus=hyperparameters.num_cpus, num_gpus=hyperparameters.num_gpus, num_return_vals=3)
    def _optimize_ppo(network_descriptions, actor_weights, critic_weights, data, hyperparameters):
        optimizer = ppo.PPO_optimizer(network_descriptions, actor_weights, critic_weights, data, hyperparameters)
        actor_weights, critic_weights, logger = optimizer.optimize()
        return actor_weights, critic_weights, logger
    ret = ray.get(_optimize_ppo.remote(network_descriptions, actor_weights, critic_weights, data, hyperparameters))
    actor_weights, critic_weights, logger = ret
    return actor_weights, critic_weights, logger
