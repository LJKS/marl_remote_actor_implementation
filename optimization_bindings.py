import ppo
import ray
"""
binding to ppo optimizer class
"""
@ray.remote(num_cpus=32, num_gpus=1, num_return_vals=3)
def optimize_ppo(network_descriptions, actor_weights, critic_weights, data):
    optimizer = ppo.PPO_optimizer(network_descriptions, actor_weights, critic_weights, data)
    actor_weights, critic_weights, logger = optimizer.optimize()
    return actor_weights, critic_weights, logger
