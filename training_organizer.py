import ray
import models
import ppo
import remote_runner
import numpy as np
import statistics

ray.init(log_to_driver=False)
#ray.init()
class Training_organizer:
    def __init__(self, steps, gym, network_descriptions, curriculum_designer):
        self.num_remotes = 8
        self.min_num_runs_generated  = 100
        self.gam = 0.99
        self.lam = 0.95
        self.finish_runs_time = 5.

        self.steps = steps
        self.gym = gym
        self.network_descriptions = network_descriptions
        self.actor_model = network_descriptions['actor'][0]
        self.critic_model = network_descriptions['critic'][0]
        self.curriculum_designer = curriculum_designer
        self.actor_weights = []
        #get actor and critic weights
        actor = self.actor_model(self.network_descriptions['actor'][1], self.network_descriptions['actor'][2], self.network_descriptions['actor'][3], self.network_descriptions['actor'][4], self.network_descriptions['actor'][5])
        self.actor_weights.append(actor.get_weights())
        critic = self.critic_model(self.network_descriptions['critic'][1], self.network_descriptions['critic'][2], self.network_descriptions['critic'][3])
        self.critic_weights = critic.get_weights()
        self.curriculum_designer = curriculum_designer

    def train(self):
        for _ in range(self.steps):
            self.training_iteration()

    def training_iteration(self):
        data = self.create_training_data()
        optimizer = ppo.PPO_optimizer.remote(self.network_descriptions, self.actor_weights[-1], self.critic_weights, data)
        new_actor_weights, new_critic_weights = ray.get(optimizer.optimize.remote())
        self.actor_weights.append(new_actor_weights)
        critic_weights = new_critic_weights


    def create_training_data(self):
        states_aggregator = []
        actions_aggregator = []
        action_log_probs_aggregator = []
        value_estimates_aggregator = []
        value_targets_aggregator = []
        advantages_aggregator = []

        sampling_results = [[] for _ in self.actor_weights]
        remote_runner_list = []
        for i in range(self.num_remotes):
            sampled_opponent = self.curriculum_designer.sample_opponent()
            remote_runner_list.append(remote_runner.Remote_Runner.remote(i, sampled_opponent, self.gym, self.network_descriptions, self.actor_weights[-1], self.critic_weights, self.actor_weights[sampled_opponent], self.gam, self.lam))
        object_ids = [remote_runner.generate_sample.remote() for remote_runner in remote_runner_list]
        episodes_generated = 0
        while episodes_generated < self.min_num_runs_generated:
            print('debug: episodes_generated', episodes_generated)
            list_done, list_not_done = ray.wait(object_ids)
            episodes_generated += len(list_done)
            finished_runs = ray.get(list_done)
            for run_data in finished_runs:
                #runner_id is the index in the remote_runner_list list, opponent_id is the index of the opponent network in in the actor_weights list
                states, actions, action_log_probs, value_targets, advantages, agent_won, runner_id, opponent_id = run_data

                states_aggregator = states_aggregator + states
                actions_aggregator = actions_aggregator + actions
                action_log_probs_aggregator = action_log_probs_aggregator + action_log_probs
                value_targets_aggregator = value_targets_aggregator + value_targets
                advantages_aggregator = advantages_aggregator + advantages

                sampled_opponent = self.curriculum_designer.sample_opponent()
                network_sampled_opponent = self.actor_weights[sampled_opponent]

                remote_runner_list[runner_id] = remote_runner.Remote_Runner.remote(runner_id, sampled_opponent, self.gym, self.network_descriptions, self.actor_weights[-1], self.critic_weights, self.actor_weights[sampled_opponent], self.gam, self.lam)

                list_not_done.append(remote_runner_list[runner_id].generate_sample.remote())
                sampling_results[opponent_id].append(agent_won)

            object_ids = list_not_done

        #finnish some more episodes
        list_done, list_not_done = ray.wait(object_ids, len(list_not_done), self.finish_runs_time)
        episodes_generated += len(list_done)
        finished_runs = ray.get(list_done)
        for run_data in finished_runs:
            #runner_id is the index in the remote_runner_list list, opponent_id is the index of the opponent network in in the actor_weights list
            states, actions, action_log_probs, value_targets, advantages, agent_won, runner_id, opponent_id = run_data

            states_aggregator = states_aggregator + states
            actions_aggregator = actions_aggregator + actions
            action_log_probs_aggregator = action_log_probs_aggregator + action_log_probs
            value_targets_aggregator = value_targets_aggregator + value_targets
            advantages_aggregator = advantages_aggregator + advantages

            sampling_results[opponent_id].append(agent_won)

        #Data is generated, update curriculum_designer and return data
        self.curriculum_designer.update(sampling_results)
        data = {'value_targets' : value_targets_aggregator, 'states' : states_aggregator, 'actions' : actions_aggregator, 'sampling_action_log_probs' : action_log_probs_aggregator, 'advantages' : advantages_aggregator}
        return data
