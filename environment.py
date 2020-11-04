"""
Author: Ashley Brockway, ashley.brockway15@ncf.edu
Date: 9/30/20
Purpose: ....
- ClusterEnv object will be called in the main execution file to be fed into the alg. as the intitaial state space
Overview of RL Parts:

    1. Observe the environment
    2. Choose action using strategy
    3. Act according to choice
    4. Receive Reward or Penalty
    5. Learn from experience
    6. Iterate until strategy is optimal based on loss

"""

from parameters import TuneMe as pa
from job import JobGrabber as jg
import matplotlib.pyplot as plt



class ClusterEnv:
    def __init__(self, set_length):
        """
         Pull from 2 other files :
            1. the parameters of our environment
                a. These create a class based on all the stuff we want to be available
                to easily change and access
            2. jobs that are entering our environment
        """

        # Construction of empty state
        self.objs_state = pa()
        self.obs_state = self.objs_state.getGrid()

        # percent of long jobs
        pct_lj = .2
        self.reslist = ["cpu", "gpu"]
        
        # Grab some jobs
        grabber = jg(pct_lj, self.reslist)
        # jobs_profile is the values themselves, log is the string info about the jobs
        self.jobs_profile, self.jobs_log = grabber.getJobs(set_length)

        # Populate the empty state with the jobs
        self.filled, self.backlog = self.objs_state.fill(self.jobs_profile, self.obs_state)

    # TODO (@ash)
    # # given that a job has been chosed, move its resource usage into the current use columns 
    # # # job choice is a numerical value representing the spot in the job queue (in the paper's case 1-10)
    # # # because this update just works for a state update, there is no need to worry about moving information from the backlog
    def updateState(self, job_choice, last_grid):
        
        if job_choice==1: 
            start_col = int(self.objs_state.res_max_len) + 1
        else: 
            start_col = int((self.objs_state.res_max_len + 1) + (int(self.objs_state.job_res_max) * job_choice) + job_choice)
            
        end_col = int(start_col + self.objs_state.job_res_max)

        height = int((self.objs_state.time_dim * self.objs_state.res_num) + (self.objs_state.res_num - 1))
        newish_grid = last_grid 
        if newish_grid[0, 0] and newish_grid[self.objs_state.time_dim + 1, 0] == 0: 
            newish_grid[0:height, 0:(end_col-start_col)] = last_grid[0:height,start_col:end_col]
        else:
            pass
        
        for indx, res in enumerate(self.reslist): 
            place_start_col = [0 for x in range(2)]
            for index, val in enumerate(last_grid[((indx * self.objs_state.time_dim) + (1 * self.objs_state.time_dim)), start_col:end_col]):
                if val != 0: 
                    pass 
                else: 
                    place_start_col[indx] = 0 
                    break
        
        
        print(place_start_col)
            #newish_grid[0:height, ]
        """
        for resource in range(self.res_num):

                # all rows that should be filled given a job's time duration
            start_row = int((resource * self.time_dim) + (resource * 2))
                end_row = int(job[-1]) + start_row

                # get the percent of the max resource being used by this job
                res_use = self.res_max_len * job[resource]
                partial_res_use = res_use % 1
                count_full_elements = int(res_use - partial_res_use)

                if count_full_elements != 0:
                    full_val = np.repeat(1, count_full_elements)
                    prog_list = np.append(full_val, [float(partial_res_use)])
                    for row in range(start_row, end_row):
                        grid[row, start_col:(start_col + count_full_elements + 1)] = prog_list
                else:
                    prog_list = float(partial_res_use)
                    for row in range(start_row, end_row):
                        grid[row, start_col] = prog_list
        """

        pass
    
    def updateTime(self, new_grid, old_backlog):
        pass

env = ClusterEnv(set_length=70)

filed = env.filled
update = env.updateState(5, filed)
plt.matshow(filed, cmap=plt.get_cmap('gray_r'))
plt.axis('off')
plt.show()