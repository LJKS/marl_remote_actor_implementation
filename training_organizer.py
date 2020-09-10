import ray
import ppo
import remote_runner
import numpy as np
import statistics
import optimization_bindings
import model_factory
import pickle
import csv
from hyperparameters import Hyperparameters
#remove remote prints to main, initialize ray cluster

#switching prints to main process


class Training_organizer:
    """
    Training_organizer class connects trajectory sampling and optimization, implementing multiprocessing using the ray multiprocessing framework
    @params:
        - steps is the number of sampling & optimization iterations taken
        - gym is the Class of the gym to be used
        - network descriptions is a dict containing list descriptions behind keys 'actor', 'critic' and 'opponent'
        - curriculum_designer is the curriculum_designer to be used
    external use of this class is supposed to be limited to train.
    """
    def __init__(self, steps, gym, network_descriptions, curriculum_designer, hyperparameters=Hyperparameters()):
        ray.init(log_to_driver=hyperparameters.log_to_driver)
        # min num remotes should be orders of magnitude smaller than min_num_runs_generated
        self.num_remotes = 32
        self.min_num_runs_generated  = 100
        self.gam = 0.999
        self.lam = 0.97
        self.finish_runs_time = 1.
        self.hyperparameters = hyperparameters

        #number of iterations of first gathering samples, then optimizing on them to run here
        self.steps = steps
        self.gym = gym
        self.network_descriptions = network_descriptions
        self.curriculum_designer = curriculum_designer
        self.actor_weights = []
        #get actor and critic weights
        a_w, c_w = ray.get(get_initial_weights.remote(self.network_descriptions))
        self.actor_weights.append(a_w)
        self.critic_weights = c_w
        #track with training iteration this is in
        self.iteration = 0
        #build logging object for this!
        self.logger = [dict()]



    def train(self):
        """
        void method, implements training loops
        """
        if not ray.is_initialized():
            ray.init(log_to_driver = self.hyperparameters.log_to_driver)
        for i in range(self.steps):
            print('Training Episode ', self.iteration)
            self.training_iteration()
            self.save()
            self.iteration +=1
            self.hyperparameters.step += 1
        ray.shutdown()

    def training_iteration(self):
        """
        implements a single training run
        """
        data = self.create_training_data()
        new_actor_weights, new_critic_weights, report = optimization_bindings.optimize_ppo(self.network_descriptions, self.actor_weights[-1], self.critic_weights, data, self.hyperparameters)
        self.logger[-1]['critic_loss']=report['critic_summary']
        self.logger[-1]['entropy']=report['entropy_summary']
        self.logger[-1]['ppo_objective']=report['policy_summary']
        self.actor_weights.append(new_actor_weights)
        critic_weights = new_critic_weights
        self.print_logger()
        self.logger.append(dict())


    def create_training_data(self):
        """
        runs @num_remotes parallel processes each returning a finished trajectory, accumulating their information and starting a new one once finished
        some processes not finishing in time after the @min_num_runs_generated is reached might not be finished and are cancelled
        """
        states_aggregator = []
        actions_aggregator = []
        action_log_probs_aggregator = []
        value_targets_aggregator = []
        advantages_aggregator = []
        rewards_aggregator = []

        sampling_results = [[] for _ in self.actor_weights]
        remote_runner_list = []

        for i in range(self.num_remotes):
            sampled_opponent = self.curriculum_designer.sample_opponent()
            remote_runner_list.append(remote_runner.Remote_Runner.remote(i, sampled_opponent, self.gym, self.network_descriptions, self.actor_weights[-1], self.critic_weights, self.actor_weights[sampled_opponent], self.gam, self.lam))
        object_ids = [remote_runner.generate_sample.remote() for remote_runner in remote_runner_list]
        episodes_generated = 0

        #TODO: move inner part of while into function to put into finally statement
        while episodes_generated < self.min_num_runs_generated:
            list_done, list_not_done = ray.wait(object_ids)
            episodes_generated += len(list_done)
            finished_runs = ray.get(list_done)

            for run_data in finished_runs:
                #runner_id is the index in the remote_runner_list list, opponent_id is the index of the opponent network in in the actor_weights list
                states, actions, action_log_probs, value_targets, advantages, rewards, agent_won, runner_id, opponent_id = run_data
                for elem, list in zip([states, actions, action_log_probs, value_targets, advantages, rewards], [states_aggregator, actions_aggregator, action_log_probs_aggregator, value_targets_aggregator, advantages_aggregator, rewards_aggregator]):
                    list.append(elem)
                sampled_opponent = self.curriculum_designer.sample_opponent()
                network_sampled_opponent = self.actor_weights[sampled_opponent]
                remote_runner_list[runner_id] = remote_runner.Remote_Runner.remote(runner_id, sampled_opponent, self.gym, self.network_descriptions, self.actor_weights[-1], self.critic_weights, self.actor_weights[sampled_opponent], self.gam, self.lam)
                list_not_done.append(remote_runner_list[runner_id].generate_sample.remote())
                sampling_results[opponent_id].append(agent_won)

            object_ids = list_not_done

        #finnish some more episodes, which have been started so we dont throw computational time just out
        list_done, list_not_done = ray.wait(object_ids, len(list_not_done), self.finish_runs_time)
        episodes_generated += len(list_done)
        finished_runs = ray.get(list_done)

        for run_data in finished_runs:
            #runner_id is the index in the remote_runner_list list, opponent_id is the index of the opponent network in in the actor_weights list
            states, actions, action_log_probs, value_targets, advantages, rewards, agent_won, runner_id, opponent_id = run_data
            for elem, list in zip([states, actions, action_log_probs, value_targets, advantages, rewards], [states_aggregator, actions_aggregator, action_log_probs_aggregator, value_targets_aggregator, advantages_aggregator, rewards_aggregator]):
                list.append(elem)
            sampling_results[opponent_id].append(agent_won)

        states_aggregator = np.concatenate(states_aggregator)
        actions_aggregator = np.concatenate(actions_aggregator)
        action_log_probs_aggregator = np.concatenate(action_log_probs_aggregator)
        value_targets_aggregator = np.concatenate(value_targets_aggregator)
        advantages_aggregator = np.concatenate(advantages_aggregator)
        rewards_aggregator = np.concatenate(rewards_aggregator)
        self.logger[-1]['mean rewards'] = np.mean(rewards_aggregator)
        self.logger[-1]['mean sampling prob'] = np.mean(np.exp(action_log_probs_aggregator))
        self.logger[-1]['episodes generated']=episodes_generated
        self.logger[-1]['average num steps']=rewards_aggregator.shape[0]/episodes_generated
        #Data is generated, update curriculum_designer and return data
        self.curriculum_designer.update(sampling_results)
        data = {'value_targets' : value_targets_aggregator, 'states' : states_aggregator, 'actions' : actions_aggregator, 'sampling_action_log_probs' : action_log_probs_aggregator, 'advantages' : advantages_aggregator}
        return data

    def print_logger(self):
        """
        prints gathered information on iteration
        """
        for dic in self.logger:
            out=''
            for key in dic:
                out = out + '  ' + key
                if key == "episodes generated":
                    out = out + " - %4d |"%(dic[key])
                else:
                    out = out + "-%11.4f |"%(dic[key])
            print(out)

    def save(self):
        #save everything from this model
        #step_str is deprecated for now
        #step_str = '_step_' + str(self.iteration)
        with open(self.hyperparameters.training_save_path + '.pkl', 'wb') as f:
            pickle.dump(self, f)
        with open(self.hyperparameters.result_save_path + '.csv', 'a') as f:
            csv_writer = csv.writer(f)
            results = self.curriculum_designer.report()
            csv_writer.writerow(self.curriculum_designer.report())

@ray.remote(num_cpus=1, num_return_vals=2)
def get_initial_weights(network_descriptions):
    """
    initializes a single network of each kind to gatheer initial weights
    """
    import model_factory
    actor_model = model_factory.get_model(network_descriptions['actor'][0])
    critic_model = model_factory.get_model(network_descriptions['critic'][0])
    actor = actor_model(network_descriptions['actor'][1], network_descriptions['actor'][2], network_descriptions['actor'][3], network_descriptions['actor'][4], network_descriptions['actor'][5])
    critic = critic_model(network_descriptions['critic'][1], network_descriptions['critic'][2], network_descriptions['critic'][3])
    actor_weights = actor.get_weights()
    critic_weights = critic.get_weights()
    return actor_weights, critic_weights
