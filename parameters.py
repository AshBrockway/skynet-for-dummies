"""
Author: Ashley Brockway, ashley.brockway15@ncf.edu
Date: 9/30/20
Purpose: ....

Grid will look like this:
    Capacity R1           | Job Slot 1  | ... | Job Slot_queue_len |    Backlog
T1  0 0 0 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 | ... | 0 0 0 0 0 0        |       0
T2  0 0 0 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 | ... | 0 0 0 0 0 0        |       0
T3  0 0 0 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 | ... | 0 0 0 0 0 0        |       0
T4  0 0 0 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 | ... | 0 0 0 0 0 0        |       0
T5  0 0 0 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 | ... | 0 0 0 0 0 0        |       0
T6  0 0 0 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 | ... | 0 0 0 0 0 0        |       0
T7  0 0 0 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 | ... | 0 0 0 0 0 0        |       0
T8  0 0 0 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 | ... | 0 0 0 0 0 0        |       0
T9  0 0 0 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 | ... | 0 0 0 0 0 0        |       0
T10 0 0 0 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 | ... | 0 0 0 0 0 0        |       0
    0 0 0 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 | ... | 0 0 0 0 0 0                0
    Capacity R2           | Job Slot 1  | ... | Job Slot_queue_len |       0
T1  0 0 0 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 | ... | 0 0 0 0 0 0        |       0
T2  0 0 0 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 | ... | 0 0 0 0 0 0        |       0
T3  0 0 0 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 | ... | 0 0 0 0 0 0        |       0
T4  0 0 0 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 | ... | 0 0 0 0 0 0        |       0
T5  0 0 0 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 | ... | 0 0 0 0 0 0        |       0
T6  0 0 0 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 | ... | 0 0 0 0 0 0        |       0
T7  0 0 0 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 | ... | 0 0 0 0 0 0        |       0
T8  0 0 0 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 | ... | 0 0 0 0 0 0        |       0
T9  0 0 0 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 | ... | 0 0 0 0 0 0        |       0
T10 0 0 0 0 0 0 0 0 0 0 0 | 0 0 0 0 0 0 | ... | 0 0 0 0 0 0        |       0

"""
import numpy as np
import random as rand

class Tune_Me:
    def __init__(self):

        # grid creation
        self.time_dim = 10
        self.res_num = 2
        self.queue_len = 3
        self.backlog_max = 10
        self.res_max_len = 12 # for grid creation
        self.job_res_max = .5 * self.res_max_len

        # define resource capacity
        self.res_cap = 1.2 # % in use

        # define jobsets aspects
        self.set_len = 10

        # episodes
        self.num_eps = self.set_len

        # time duration
        self.long_time = .8 # distribution propoerty of "long jobs"
        self.short_time = 1 - self.long_time

        # get list of resource names
        self.res_list = ("cpu", "gpu")
        self.dom_res = rand.sample(self.res_list, 1)

        # seed
        self.start_seed = np.random.uniform(low=1, high=1000000000, size=1)
        self.new_seed = np.random.uniform(low=-45, high=45, size=1) + self.start_seed

    def render_image(self):
        self.width = self.res_max_len + (self.job_res_max * self.res_num)

    # Here I fill the jobs ............................
    def fill(self, jobs):
        # TBD lets look at jobs
