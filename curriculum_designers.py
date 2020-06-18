import statistics
import random

class Self_play:
    def __init__(self):
        self.results = []
        self.step = 0

    def sample_opponent(self):
        return self.step

    def update(self, updates):
        mean_results = [statistics.mean(idx_results) if len(idx_results)>0 else None for idx_results in updates]
        self.results.append(mean_results)
        print(mean_results)
        self.step = self.step + 1

    def report(self):
        last_step = self.results[-1]
        return last_step, statistics.mean(last_step)

class Uniform_Sampling:
    def __init__(self):
        self.results = []
        self.step = 0

    def sample_opponent(self):
        return random.randint(0,self.step)

    def update(self, updates):
        mean_results = [statistics.mean(idx_results) if len(idx_results)>0 else None for idx_results in updates]
        self.results.append(mean_results)
        print(mean_results)
        self.step = self.step + 1

    def report(self):
        return self.results[-1]

class OriginalSelf:
    def __init__(self):
        self.results = []
        self.step = 0

    def sample_opponent(self):
        return 0

    def update(self, updates):
        mean_results = [statistics.mean(idx_results) if len(idx_results)>0 else None for idx_results in updates]
        self.results.append(mean_results)
        print(mean_results)
        self.step = self.step + 1

    def report(self):
        last_step = self.results[-1]
        mean_results = [statistics.mean(idx_results) if len(idx_results)>0 else None for idx_results in last_step]
        return mean_results
