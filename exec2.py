import training_organizer
import marl_gyms
import curriculum_designers

gym = marl_gyms.LunarLanderC_POC
input_shape = gym(None).get_observation_shape()
actor_description = ['MLP_model', [64,64], 'continuous', 2, 0.0003, input_shape]
critic_description = ['V_MLP_model', [64,64], 0.001, input_shape]
opponent_description = actor_description
network_descriptions = {'actor': actor_description, 'critic': critic_description, 'opponent':opponent_description}
training = training_organizer.Training_organizer(200, gym, network_descriptions, curriculum_designers.Self_play())
training.train()
