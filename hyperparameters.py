from datetime import datetime
class Hyperparameters:
    def __init__(self, save_name=str(datetime.now()).replace(' ', '_'), step=0, log_to_driver=False, num_cpus=16, num_gpus=1, **kwargs):
        self.save_name = save_name
        self.step = step
        self.log_to_driver = log_to_driver
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self.model_save_path = 'saves/saved_models/model_' + save_name
        self.result_save_path = 'saves/saved_results/result_' + save_name
        self.training_save_path = 'saves/saved_training_states/' + save_name
