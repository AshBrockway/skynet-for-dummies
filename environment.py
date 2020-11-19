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
        self.filled = self.objs_state.fill(self.jobs_profile, self.obs_state)

    
    def updateState(self, job_choice, last_grid):
        
        if job_choice==0: 
            new_grid = last_grid
            stop = True
        else: 
            schedule_job = self.jobs_profile[job_choice]
            #print(schedule_job)
            new_grid = last_grid
        
            for resource in range(self.objs_state.res_num): 
                stop = False
                start_row = int((resource * self.objs_state.time_dim) + (resource * 2))
                end_row = int(start_row + schedule_job[0][-1])
            
                if job_choice==1:
                    old_start_col = int(self.objs_state.res_max_len) + 1
                else:
                    old_start_col = int(self.objs_state.res_max_len) + 1 +(int(self.objs_state.job_res_max) * (job_choice - 1)) + (job_choice - 1)
                
                for elem in range(int(self.objs_state.res_max_len)):
            
                    # then move forward in time
                    if last_grid[start_row,elem]==0: 
                        if (int(self.objs_state.res_max_len) - elem) < len(schedule_job[1][resource]):
                            new_grid = last_grid 
                            stop = True
                        else:
                            for row in range(start_row, end_row):
                                new_grid[row, elem:elem + len(schedule_job[1][resource])] = schedule_job[1][resource]
                                new_grid[row, old_start_col:old_start_col + int(self.objs_state.job_res_max)] = [0 for x in range(int(self.objs_state.job_res_max))] 
                
                        break
            
                if stop: 
                    break
        if stop: 
            new_grid = self.updateTime(new_grid2=new_grid) 
            # call update time
                
        return(new_grid)
    
    def updateTime(self, new_grid2):
        time_grid = new_grid2
        start_col = 0 
        end_col = int(self.objs_state.res_max_len) - 1
        
        res1_start_row = 0 
        res1_end_row = 21
        
        res2_start_row = 22
        res2_end_row = 40
        
        time_grid[res1_start_row:res1_end_row - 1, start_col:end_col] = new_grid2[res1_start_row + 1:res1_end_row, start_col:end_col]
        
        time_grid[res2_start_row:res2_end_row - 1, start_col:end_col] = new_grid2[res2_start_row + 1: res2_end_row, start_col:end_col]
        
        return(time_grid)

env = ClusterEnv(set_length=70)

grid = env.filled

new = env.updateState(8, grid)

#newer = env.updateState(0, new)

plt.matshow(new, cmap=plt.get_cmap('gray_r'))
plt.axis('off')
plt.show()