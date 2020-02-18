import training_organizer
import marl_gyms
import curriculum_designers
import models

gym = marl_gyms.Bet_gym
input_shape = gym(None).get_observation_shape()
actor_description = [models.MLP_model, [32,32], 'continuous', 9, 0.001, input_shape]
critic_description = [models.V_MLP_model, [32,32], 0.001, input_shape]
opponent_description = actor_description
network_descriptions = {'actor': actor_description, 'critic': critic_description, 'opponent':opponent_description}
training = training_organizer.Training_organizer(100, gym, network_descriptions, curriculum_designers.Self_play())
training.train()
