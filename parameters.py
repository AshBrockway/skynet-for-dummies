"""
Authors: Ashley Brockway, ashley.brockway15@ncf.edu, Jamie Hobbs, jamie.hobbs19@ncf.edu
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

The initial object will be a matrix of 0s, including 0 padding between rows for different resources and columns of different jobs

The fill method will fill the grid with the job profiles from the jobs.py file #
"""
import numpy as np
from job import JobGrabber as jg
import matplotlib.pyplot as plt
from DBconnect import *


class TuneMe:
    """
    The point of this object is to store aspects of the system that can be tweaked for different training iterations.
    The basis of the parameters are to help form the basis for our environmental representation shown in the doc string at the beginning of this file
    """
    def __init__(self):

        ##The below parameters are tuneable. Update with care!!
        # empty grid creation

        self.time_dim = 15
        # number of jobs in queue, or the amount of jobs that are rendering in the observed state (1 less than expected python stuff)
        self.queue_len = 6
        # resource capacities shown in the grid, think of filling these as usage_proportion*12
        self.res_max_len = 12
        # .5 units of resource is the max % of resource used by one job in current sample
        self.job_res_max = 6

        ##The below parameters are not tuneable and more for information purposes
        # define jobsets aspects
        self.set_len = 10
        # The largest number of jobs that could be in the backlog
        self.backlog_max = 10

        #--------------------DASHBOARD Parameters--------------------#
        ##The below parameters are tuneable and are displayed on the dashboard.
        # proportion of long jobs
        self.long = float(.5)
        # number of resources in problem
        self.res_num = 2
        #number of jobs
        self.jobs = 16

        ##The below parameters are not tuneable at this moment, but if updated will change the display on the dashboard.
        # proportion of short jobs
        self.short = float(1 - self.long)
        # define resource capacity
        self.res_cap = 1.0# % in use
        # resource usage low value
        self.reslow = .05
        # resource usage high value
        self.reshigh = .5
        # time usage low value
        self.timelow = 1
        # time usage high value
        self.timehigh = 15
        # -----------------------------------------------------------#

        # seed to be used to set seed for run overall
        self.start_seed = np.random.uniform(low=1, high=1000000000, size=1)
        self.new_seed = np.random.uniform(low=-45, high=45, size=1) + self.start_seed

    def updateDB(self, to_update):
        if to_update == 'params':
            print("insides updateDB function in params")
            values = []
            values.extend((self.res_num, self.res_cap, self.long, self.short, self.reslow, self.reshigh, self.timelow, self.timehigh, self.jobs))
            print(values)
            # connect to database to update parameters table
            DBconnect.updateparams(values)

    def getLong(self):
        long = self.long
        return long

    def getNumJobs(self):
        jobs = self.jobs
        return jobs

    def getGrid(self):
        # take the number of rows with time_dim times the number of resources as well as one row to pad between resrouces given that the is >1 resources
        height = int((self.time_dim * self.res_num) + (self.res_num - 1))
        # add the max resource length, the number of jobs in queue times the number of squares in the resources row plus
        # # the number of jobs in the queues (i.e. the columns between jobs) and 2 more colunms for the backlog and white space in between
        width = int(self.res_max_len + (self.queue_len * self.job_res_max) + self.queue_len + 2)

        # create the shell of one column of the grid (basically a list of 0s the length of the height calculated above)
        empty_grid = [0 for x in range(height)]
        # For each row, laid out in the shell line above, fill out 0s across the columns whose dimension is set with the width calculation above
        for i in range(height):
            # creates a list of 0s using the width of the grid
            empty_grid[i] = [0 for x in range(width)]

        # makes a list of lists into a matrix
        empty_grid = np.array(empty_grid, dtype=float)
        return(empty_grid)

    # this method is to fill the empty grid created with the getGrid method of this file with jobs we generate using the JobGrabber in the jobs file
    # # The filled grid takes a jobset and a empty grid as
    
    def fill(self, jobs, empty_grid):
        # subset the job list to only include the jobs for subset M (length defined above)
        for kye, val in jobs.items():
            p_list = self.getProgList(val)
            jobs[kye].append(p_list)
        
        self.jobs_subset = {key1: value1 for key1, value1 in jobs.items() if int(key1) < int(self.queue_len) + 1}
        self.backlog_subset = {keyb: valueb for keyb, valueb in jobs.items() if int(keyb) > int(self.queue_len)}
        
        height = (int(self.time_dim) * int(self.res_num)) + (int(self.res_num) - 1)
        grid = empty_grid.astype(float)
        
        # Creation of dictionary to hold job information 
        
        """
            The Grid must be filled. Components of this filling are:
                1. Backlog count
                2. For each Resource:
                    a. We will not have to fill any info into the resource current usage column, Because
                        fill will only be used to fill the initial observed state, placing the jobs in their positions for training
                    b. For each Job in queue:
                        i. Fill out the % usage in the # of rows associated with their time
        """
        # Now start with the easy part, place the # of jobs in the backlog in the last column of the grid
        if len(self.backlog_subset.keys()) < height: 
            grid[0:len(self.backlog_subset.keys()), -1] = [1 for x in range(self.backlog_subset.keys())]
        else:
            grid[0:(height), -1] = [1 for x in range(height)]

        
        job_count = -1

        for job, infor in self.jobs_subset.items():
            job_count += 1

            # using the # job being visualized, we find the starting index of the grid, given our rows, to start filling at
            if job_count < 1:
                start_col = int(self.res_max_len) + 1

            else:
                start_col = int(self.res_max_len) + 1 + (int(self.job_res_max) * job_count) + job_count


            for resource in range(self.res_num):
                end_col = len(infor[1][resource]) + start_col
                # all rows that should be filled given a job's time duration
                start_row = int((resource * self.time_dim) + (resource * 2))
                end_row = int(infor[0][-1]) + start_row

                for row in range(start_row, end_row):
                    grid[row, start_col:end_col] = infor[1][resource]
               
               
        return(grid)

    def getProgList(self, jobi):
        
        job_prog_list = [[0] for x in range(self.res_num)]
        
        for res in range(self.res_num): 
            res_use = float(self.res_max_len) * jobi[0][int(res)]
            partial_use = res_use % 1 
            count_full_elements = float(res_use - partial_use)
            
            full_val = np.repeat(1, count_full_elements)
            if partial_use != 0: 
                prog_list = np.append(full_val, [float(partial_use)])
            else: 
                prog_list = full_val
            
            job_prog = list(prog_list)
        
            job_prog_list[res] = job_prog
        return(job_prog_list)
# Testing
"""
pa = TuneMe()
emptyGrid = pa.getGrid()
grabber = jg(.2, ['cpu', 'gpu'])
jobsset, jobs_log = grabber.getJobs(set_num=70)
ggrid = pa.fill(jobsset, emptyGrid)

plt.matshow(ggrid, cmap=plt.get_cmap('gray_r'))
plt.axis('off')


#plt.show()
# print(emptyGrid)
# print(emptyGrid.shape)

#print(pa.res_num, pa.res_cap, pa.long, pa.short, pa.reslow, pa.reshigh, pa.timelow, pa.timehigh)
# values = []
# values.extend((pa.res_num, pa.res_cap, pa.long, pa.short, pa.reslow, pa.reshigh, pa.timelow, pa.timehigh))
# print(values)
#connect to database to update parameters
#DBconnect.updateparams(values)

