import statistics


class Self_play:
    def __init__(self):
        self.results = []
        self.step = 0

    def sample_opponent(self):
        return self.step

    def update(self, updates):
        self.results.append(updates)
        #debug only
        results = [statistics.mean(step_results[0]) for step_results in self.results]
        print(results)
        
    def report(self):
        last_step = self.results[-1]
        mean_results = [statistics.mean(idx_results) if len(idx_results)>0 else None for idx_results in last_step]
        return mean_results

class Uniform_Sampling:
    def __init__(self, fraction):
        self.results = []
        self.step = 0
    def sample_opponent(self):
        a = 5
        ## TODO:

    def update(self, updates):
        a=5
        ## TODO:
    def report(self):
        a=5
        #TODO
