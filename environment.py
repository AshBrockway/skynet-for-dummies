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
"""

from parameters import TuneMe as pa
from job import JobGrabber as jG
import matplotlib.pyplot as plt
import numpy as np



class ClusterEnv:
    def __init__(self, set_length):
        

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
        self.backlog_keys = self.objs_state.backlog_subset.keys() 
        self.que_keys = self.objs_state.jobs_subset.keys() 
        self.past_jobs = {}
        self.choice_list = []
        self.count  = 0
        self.rewards = {} 
        
        for keys in self.que_keys: 
                self.objs_state.jobs_subset[keys].append([keys])
            
    
    def updateState(self, job_choice, last_grid):
        
        stop = False 
        # given that the void action is chosen we just keep the same grid and move on in time
        if job_choice==0: 
            new_grid = last_grid
            stop = True
        else: 
            
            # our schedule job consists of a list of information about that job form the job queue 
            schedule_job = self.objs_state.jobs_subset[job_choice]
           
            self.past_jobs[schedule_job[2][0]] = schedule_job 
            
            
            # we start by setting the new grid equal to the last grid to have a skeleton to work with 
            new_grid = last_grid
            
            if job_choice==1:
                old_start_col = 13
            else:
                old_start_col = int(self.objs_state.res_max_len) + 1 +(int(self.objs_state.job_res_max) * (job_choice - 1)) + (job_choice - 1)
            
            self.past_jobs[schedule_job[2][0]].append([old_start_col])
            
            
             # for each resrouces profile list 
            for resource in range(self.objs_state.res_num): 
                
                # get the rows 
                start_row = int((resource * self.objs_state.time_dim) + (resource * 2))
                end_row = int(start_row + schedule_job[0][-1])
                
                for elem in range(int(self.objs_state.res_max_len)):
                    # then move forward in time
                    if last_grid[start_row,elem]==0: 
                        if (int(self.objs_state.res_max_len) - elem) < len(self.past_jobs[job_choice][1][resource]):
                            stop = True
                            break 
                        else: 
                            if job_choice==1:
                                old_start_col = int(self.objs_state.res_max_len) + 1
                            else:
                                old_start_col = int(self.objs_state.res_max_len) + 1 + (int(self.objs_state.job_res_max) * (job_choice - 1)) + (job_choice - 1)
                    
                            for row in range(start_row, end_row):
                                new_grid[row, elem:elem + len(schedule_job[1][resource])] = schedule_job[1][resource]
                                new_grid[row, old_start_col:old_start_col + int(self.objs_state.job_res_max)] = [0 for x in range(int(self.objs_state.job_res_max))] 
                            break 
                    
            if stop: 
                new_grid = last_grid 
                del self.past_jobs[schedule_job[2][0]]
            
        # feed the last job choice to the choice list even if that choice was a void job or a break given that the break rewrote the choice 
        self.choice_list.append(job_choice)
        
        # stop will be true when there is a void action or when the agen choses the void action 
        if stop: 
            # update time 
            new_grid = self.updateTime(new_grid2=new_grid)
            # calculate rewards 
            self.rewards[self.count]= self.getReward(self.choice_list)
            #self.rewards[]
            # erase past choices after time steps and rewards are finished 
            self.choice_list = []
            

        return(new_grid)
    
    def updateTime(self, new_grid2):
        self.count += 1
        self.taken = {}
        time_grid = self.obs_state
        
        time_grid[0:18, 0:11] = new_grid2[1:19, 0:11]
        
        time_grid[22:39, 0:11] = new_grid2[23:40, 0:11]
        time_grid[0:40, 12:84] = new_grid2[0:40, 12:84]

        mid_grid = self.shiftCurrent(time_grid) 
        
        moved_grid = mid_grid
        
        c = 1 
        for i in self.choice_list:
            if i==0:
                del self.choice_list[c-1]
            else: 
                moved_grid = self.moveFromBack(mid_grid, i, c)
                
            c += 1 
        if len(self.objs_state.backlog_subset.keys()) > 41 :
            pass
        else: 
            moved_grid[41 - len(self.choice_list) - 1: 41, -1] = [0 for vali in range(42-len(self.choice_list), 41)]
        
        return(moved_grid)

    def moveFromBack(self, tg, jc, cn):
        moved_stuff = tg
        newie = self.objs_state.backlog_subset[(len(self.past_jobs.keys()) + 10) + cn]
        nk = (len(self.past_jobs.keys()) + 10)
        del self.objs_state.jobs_subset[jc]
        
        self.objs_state.jobs_subset[jc] = newie
        self.objs_state.jobs_subset[jc].append([nk])
        
     
        empty_job_start_col = int(self.objs_state.res_max_len) + 1 +(int(self.objs_state.job_res_max) * (jc - 1)) + (jc - 1)
        
        end_col_r1 = empty_job_start_col + len(self.objs_state.jobs_subset[jc][1][0])
        end_col_r2 = empty_job_start_col + len(self.objs_state.jobs_subset[jc][1][1])
        
        time = int(self.objs_state.jobs_subset[jc][0][-1])
        
        for r in range(time):
            moved_stuff[r, empty_job_start_col:end_col_r1] = np.array(self.objs_state.jobs_subset[jc][1][0])
            moved_stuff[22 + r, empty_job_start_col:end_col_r2] = np.array(self.objs_state.jobs_subset[jc][1][1])
        
        return(moved_stuff)
    
    def shiftCurrent(self, tg): 
        ng = tg
        
        
        val_r1 = tg[0:20, 0:11]
        val_r2 = tg[21:41, 0:11]
        
        
        
        for row1 in range(20): 
            n_0s1 = [] 
            n_val1 = []
            
            for v1 in val_r1[row1,0:11]: 
                if v1==0: 
                    n_0s1.append(v1)
                else: 
                    n_val1.append(v1)
            
            ng[row1, 0:len(n_val1)] = n_val1
            ng[row1, len(n_val1):11] = n_0s1
        
        
        for row2 in range(20): 
            n_0s2 = [] 
            n_val2 = []
            
            for v2 in val_r2[row2,0:11]: 
                if v2==0: 
                    n_0s2.append(v2)
                else: 
                    n_val2.append(v2)
                
            
            ng[row2 + 21, 0:len(n_val2)] = n_val2
            ng[row2 + 21, len(n_val2):11] = n_0s2
            
          
        return(ng)
        
    def getReward(self, choices): 
        taken = []
        
        for i in choices: 
            taken.append(self.count + self.jobs_profile[i][0][-1])
        
        print(taken)
        return(taken)

env = ClusterEnv(set_length=70)

grid = env.filled

new = env.updateState(8, grid)

"""
new2 = env.updateState(2, new)

new3 = env.updateState(0, new2)

new4 = env.updateState(3, new3) 

new5 = env.updateState(4, new4) 

new6 = env.updateState(9, new5) 

new7 = env.updateState(0, new6) 

new8 = env.updateState(9, new7) 

new9 = env.updateState(3, new8) 

new10 = env.updateState(6, new9) 


new11 = env.updateState(0, new10) 

new12 = env.updateState(5, new11) 

new13 = env.updateState(8, new12) 


new14 = env.updateState(1, new13) 


new15 = env.updateState(0, new14) 
"""


"""
plt.matshow(new, cmap=plt.get_cmap('gray_r'))
plt.axis('off')
plt.show()
"""