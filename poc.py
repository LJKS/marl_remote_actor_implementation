import training_organizer
import marl_gyms
import curriculum_designers
import vier_gewinnt
import snake_gym
import tron_gym
import hyperparameters

mode = 'Multi_Tron'
if mode == 'discrete':
    gym = marl_gyms.LunarLander_POC
    input_shape = gym(None).get_observation_shape()
    actor_description = ['MLP_model', [32,32], 'discrete', 4, 0.0003, input_shape]
    critic_description = ['V_MLP_model', [32,32], 0.001, input_shape]
    opponent_description = actor_description
    network_descriptions = {'actor': actor_description, 'critic': critic_description, 'opponent':opponent_description}
    training = training_organizer.Training_organizer(200, gym, network_descriptions, curriculum_designers.Self_play())
    training.train()
elif mode == 'continuous':
    gym = marl_gyms.LunarLanderC_POC
    input_shape = gym(None).get_observation_shape()
    actor_description = ['MLP_model', [32,32], 'continuous', 2, 0.0003, input_shape]
    critic_description = ['V_MLP_model', [32,32], 0.001, input_shape]
    opponent_description = actor_description
    network_descriptions = {'actor': actor_description, 'critic': critic_description, 'opponent':opponent_description}
    training = training_organizer.Training_organizer(200, gym, network_descriptions, curriculum_designers.Self_play())
    training.train()
elif mode == 'Self_play':
    gym = vier_gewinnt.Vier_gewinnt_gym
    input_shape = gym(None).get_observation_shape()
    actor_description = ['MLP_model', [32,32], 'discrete', 7, 0.0003, input_shape]
    critic_description = ['V_MLP_model', [32,32], 0.001, input_shape]
    opponent_description = actor_description
    network_descriptions = {'actor': actor_description, 'critic': critic_description, 'opponent':opponent_description}
    training = training_organizer.Training_organizer(200, gym, network_descriptions, curriculum_designers.Self_play())
    training.train()
elif mode == 'Uniform':
    gym = vier_gewinnt.Vier_gewinnt_gym
    input_shape = gym(None).get_observation_shape()
    actor_description = ['MLP_model', [128,128,32], 'discrete', 7, 0.0003, input_shape]
    critic_description = ['V_MLP_model', [128,128,32], 0.001, input_shape]
    opponent_description = actor_description
    network_descriptions = {'actor': actor_description, 'critic': critic_description, 'opponent':opponent_description}
    training = training_organizer.Training_organizer(200, gym, network_descriptions, curriculum_designers.Uniform_Sampling())
    training.train()
elif mode == 'OriginalSelf':
    gym = vier_gewinnt.Vier_gewinnt_gym
    input_shape = gym(None).get_observation_shape()
    actor_description = ['MLP_model', [128,64,32], 'discrete', 7, 0.0003, input_shape]
    critic_description = ['V_MLP_model', [128,64,32], 0.001, input_shape]
    opponent_description = actor_description
    network_descriptions = {'actor': actor_description, 'critic': critic_description, 'opponent':opponent_description}
    training = training_organizer.Training_organizer(200, gym, network_descriptions, curriculum_designers.OriginalSelf())
    training.train()
elif mode=='Snake_gym':
    gym = snake_gym.Snake_gym
    action_size = gym(None).get_action_size()
    input_shape = gym(None).get_observation_shape()
    cnn_dict = {'layers':15, 'filters':4, 'mlp':[1024,128]}
    actor_description = ['Dense_CNN_Model', cnn_dict, 'discrete', action_size, 0.0003, input_shape]
    critic_description = ['V_Dense_CNN_Model', cnn_dict, 0.001, input_shape]
    opponent_description = actor_description
    network_descriptions = {'actor': actor_description, 'critic': critic_description, 'opponent':opponent_description}
    training = training_organizer.Training_organizer(200, gym, network_descriptions, curriculum_designers.Self_play())
    training.train()
elif mode=='Tron_gym':
    gym = tron_gym.Tron_gym
    action_size = gym(None).get_action_size()
    input_shape = gym(None).get_observation_shape()
    cnn_dict = {'layers':5, 'filters':4, 'mlp':[1024,128]}
    actor_description = ['Dense_CNN_Model', cnn_dict, 'discrete', action_size, 0.0003, input_shape]
    critic_description = ['V_Dense_CNN_Model', cnn_dict, 0.001, input_shape]
    opponent_description = actor_description
    network_descriptions = {'actor': actor_description, 'critic': critic_description, 'opponent':opponent_description}
    training = training_organizer.Training_organizer(200, gym, network_descriptions, curriculum_designers.Uniform_Sampling())
    training.train()
elif mode == 'Multi_Tron':
    for _ in range(10):
        gym = tron_gym.Tron_gym
        action_size = gym(None).get_action_size()
        input_shape = gym(None).get_observation_shape()
        cnn_dict = {'layers':5, 'filters':4, 'mlp':[1024,128]}
        actor_description = ['Dense_CNN_Model', cnn_dict, 'discrete', action_size, 0.0003, input_shape]
        critic_description = ['V_Dense_CNN_Model', cnn_dict, 0.001, input_shape]
        opponent_description = actor_description
        network_descriptions = {'actor': actor_description, 'critic': critic_description, 'opponent':opponent_description}
        training = training_organizer.Training_organizer(200, gym, network_descriptions, curriculum_designers.Uniform_Sampling(), hyperparameters=hyperparameters.Hyperparameters())
        training.train()
