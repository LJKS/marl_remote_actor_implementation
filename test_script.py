import training_organizer
import pickle
#load own training run
own_path = 'saves/saved_training_states/2020-08-03_00:19:38.545881.pkl'
own = pickle.load(open(own_path, 'rb'))
print(own.gym)
#list of paths to test against
test_paths = [own_path]
results = own.test_against(test_paths, num_tests=5, num_cpus=8, mode=[0,10,20])
result_path = 'saves/saved_results/'
save_name = '2020-08-03_00:19:38.545881_self_evaluation'
pickle.dump( results, open( result_path+save_name, "wb" ) )
