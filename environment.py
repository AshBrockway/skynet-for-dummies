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
from job import JobGrabber as jG
import matplotlib.pyplot as plt
import numpy as np



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
        grabber = jG(pct_lj, self.reslist)
        # jobs_profile is the values themselves, log is the string info about the jobs
        self.jobs_profile, self.jobs_log = grabber.getJobs(set_length)

        # Populate the empty state with the jobs
        self.filled = self.objs_state.fill(self.jobs_profile, self.obs_state)
        self.past_jobs = {}
        self.choice_list = []
        self.count = 0
        self.scheds = 0
    
    def updateState(self, job_choice, last_grid):
        if job_choice==0: 
            new_grid = last_grid
            stop = True
        else: 
            self.scheds += 1
            schedule_job = self.jobs_profile[job_choice]
            self.past_jobs[job_choice] = schedule_job
            #print(schedule_job)
            new_grid = last_grid
        
            for resource in range(self.objs_state.res_num): 
                stop = False
                start_row = int((resource * self.objs_state.time_dim) + (resource * 2))
                end_row = int(start_row + schedule_job[0][-1])
            
                if job_choice==1:
                    old_start_col = int(self.objs_state.res_max_len) + 1
                    self.past_jobs[job_choice].append([old_start_col])
                else:
                    old_start_col = int(self.objs_state.res_max_len) + 1 +(int(self.objs_state.job_res_max) * (job_choice - 1)) + (job_choice - 1)
                    self.past_jobs[job_choice].append([old_start_col])
                
                for elem in range(int(self.objs_state.res_max_len)):
            
                    # then move forward in time
                    if last_grid[start_row,elem]==0: 
                        if (int(self.objs_state.res_max_len) - elem) < len(schedule_job[1][resource]):
                            new_grid = last_grid 
                            job_choice = "B" 
                            stop = True
                        else:
                            for row in range(start_row, end_row):
                                new_grid[row, elem:elem + len(schedule_job[1][resource])] = schedule_job[1][resource]
                                new_grid[row, old_start_col:old_start_col + int(self.objs_state.job_res_max)] = [0 for x in range(int(self.objs_state.job_res_max))] 
                                
                
                        break
                
                if stop: 
                    break
                
        self.choice_list.append(job_choice)
        if stop: 
         
            self.job_choi = self.choice_list[-1]
            self.choice_list.append("T")
            new_grid = self.updateTime(new_grid2=new_grid) 
            self.count += 1
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
        
        #time_grid = self.shiftCurrent(time_grid)
        
        stp = 1
        for i in self.choice_list:
            choice = i
            if self.count < 2: 
                if choice=="T" or choice=="B" or choice==0:
                    pass
                else: 
                    moved_grid = self.moveFromBack(time_grid, i)
            else: 
                if choice !="T" or choice != "B" or choice==0:
                    pass
                else: 
                    stp += 1 
                    if stp < self.count: 
                        pass 
                    else: 
                        moved_grid = self.moveFromBack(time_grid, i)
           
        
        
        if len(self.objs_state.backlog_subset.keys()) > 41 :
                pass
        else: 
            moved_grid[41 - stp - 1: 41, -1] = [0 for vali in range(41-stp-1, 41)]
        return(moved_grid)

    def moveFromBack(self, tg, jc):
        moved_stuff = tg
        newie = self.objs_state.backlog_subset[self.scheds + self.objs_state.queue_len]
        del self.objs_state.jobs_subset[jc]
        
        self.objs_state.jobs_subset[jc] = newie
        del self.objs_state.backlog_subset[self.scheds + self.objs_state.queue_len]
        
        empty_job_start_col = self.past_jobs[jc][2][0]
        times = int(self.objs_state.jobs_subset[jc][0][-1])
               
        res1_new_start_row = 0 
        res1_new_end_row = times 
            
        res2_new_start_row = 22
        res2_new_end_row = 22 + times 
            
        
        resources = [0, 1]
        
        
        end_col_r1 = empty_job_start_col + len(self.objs_state.jobs_subset[jc][1][0])
        end_col_r2 = empty_job_start_col + len(self.objs_state.jobs_subset[jc][1][1])
        
        stuff = [[res1_new_start_row,res1_new_end_row, end_col_r1], [res2_new_start_row,res2_new_end_row, end_col_r2]]
        
        for res in resources: 
            sr = stuff[res*1][0] 
        
            for t in range(times): 
                  moved_stuff[sr + t, empty_job_start_col:stuff[res][2]] = np.array(self.objs_state.jobs_subset[jc][1][res])
                  
        
        return(moved_stuff)
    
    def shiftCurrent(self, tg): 
        pass
env = ClusterEnv(set_length=70)

grid = env.filled

new = env.updateState(8, grid)


newer = env.updateState(0, new)
new2 = env.updateState(2, newer)
plt.matshow(new2, cmap=plt.get_cmap('gray_r'))
plt.axis('off')
plt.show()