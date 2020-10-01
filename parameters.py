"""
Author: Ashley Brockway, ashley.brockway15@ncf.edu
Date: 9/30/20
Purpose: The purpose of TuneMe is to create an object where we can access different tuneable parameters of our environment.

Description:
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

The itial object will be a matrix of 0s, including 0 padding between rows for different resources and columns of different jobs

The fill method will fill the grid with the job profiles from the jobs.py file #
"""
import numpy as np
import random as rand

class TuneMe:
    def __init__(self):

        # empty grid creation
        self.time_dim = 10
        # number of resources in problem
        self.res_num = 2
        # number of jobs in queue, or the amount of jobs that are rendering in the observed state
        self.queue_len = 3
        # The largest number of jobs that could be in the backlog
        self.backlog_max = 10
        # resource capacities shown in the grid, think of filling these as usage_proportion*12
        self.res_max_len = 12

        # .5 units of resource is the max % of resource used by one job in current sample
        self.job_res_max = .5 * self.res_max_len

        # define resource capacity
        self.res_cap = 1.0# % in use

        # define jobsets aspects
        self.set_len = 10

        # episodes
        self.num_eps = self.set_len

        # seed to be used to set seed for run overall
        self.start_seed = np.random.uniform(low=1, high=1000000000, size=1)
        self.new_seed = np.random.uniform(low=-45, high=45, size=1) + self.start_seed

    def getGrid(self):
        # take the number of rows with time_dim and add an row for each resource
        height = int(self.time_dim + (1 * self.res_num) - 1)
        # add the max resource length, the number of jobs in queue times the number of squares in the resources row plus
        # # the number of jobs in the queues (i.e. the columns between jobs) and 2 more colunms for the backlog and white space in between
        width = int(self.res_max_len + (self.queue_len * self.job_res_max) + self.queue_len + 1)

        # create a list of lists created from 0s
        empty_grid = [0 for x in range(height)]
        for i in range(height):
            # creates a list of 0s using the width of the grid
            empty_grid[i] = [0 for x in range(width)]

        # makes a list of lists into a matrix
        empty_grid = np.array(empty_grid)
        return(empty_grid)

    # Here I fill the jobs ............................
    # TODO (@ash) create iterative method to place values in Grid elements keeping padding in mind
    def fill(self, jobs):
        jobs_subset = jobs[:self.queue_len]
        return(jobs_subset)

# pa = TuneMe()

# emptyGrid = pa.getGrid()
# print(emptyGrid)
# print(emptyGrid.shape)
