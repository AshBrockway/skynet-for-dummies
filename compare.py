"""
Author: Andrew Reilly, andrew.reilly19@ncf.edu
Date: 10/10/20
Purpose: This file contains multiple classses, outlining each of the comparison methods from the DeepRM paper:
    1. Shortest Job First - allocate jobs based on the job duration
    2. Packer - Allocate jobs based on resource availability
    3. Random - Picks jobs at random (used in original paper)
    4. FIFO - Requested by Novetta, simply allocates jobs based on the order they arrive
    5. ‘Tetris’ - An pseudo-intelligent agent that creates a combined score between job duration and resource availability and allocates jobs
        based on that score.
    The idea for these classes is to function similar to the DPN - at each step, select what jobs to process, send it back to the environment to process
    the next step. The main difference is that this won't take an image as input, but instead they will simply take a list of jobs
TODO: Fill in additional comments
"""

import pandas as pd
import numpy as np
import random as rand
from job import JobGrabber as jg 

class compare_models:
    def __init__(self, job_dict, res_num, res_cap):
        #pulling in necessary parameters
        n_jobs = len(job_dict)
        model_functions = ['FIFO','SJF','Random','Packer','Tetris']

        #transforming job dictionary into dataframe
        jobs_df = unpack_dict(job_dict, n_res=res_num)

        #transforms df to include slowdown information for every job on the dataframe
        jobs_df, completion_dict = process_df(jobs_df, job_dict, model_functions, res_num, n_jobs, res_cap)
        self.full_jobs_df = jobs_df

        #slowdown variables for later storage
        self.FIFO_loss = jobs_df['FIFO_slowdown'].mean()
        self.SJF_loss = jobs_df['SJF_slowdown'].mean()
        self.Random_loss = jobs_df['Random_slowdown'].mean()
        self.Packer_loss = jobs_df['Packer_slowdown'].mean()
        self.Tetris_loss = jobs_df['Tetris_slowdown'].mean()

        #alternative reward
        self.FIFO_actime = completion_dict.get('FIFO')
        self.SJF_actime = completion_dict.get('SJF')
        self.Random_actime = completion_dict.get('Random')
        self.Packer_actime = completion_dict.get('Packer')
        self.Tetris_actime = completion_dict.get('Tetris')


"""
Primary function
"""
#processes jobs_df to include comparison models
def process_df(jobs_df, job_dict, model_functions, res_num, n_jobs, res_cap):
    completion_dict = {}
    for m_function in model_functions:
        #initializing time array to track jobs/slowdown, other inits
        states = time_array(n_jobs*2, res_num)
        time_pointer = 1
        job_slowdown = {}
        jobs_left = []

        #copying job dict for use in loop
        poss_jobs = job_dict.copy()
        poss_jobs_df = jobs_df
        #looping through to assign jobs based on the model
        while len(poss_jobs) > 0:
            last_time_pointer = time_pointer
            res_available = states[time_pointer,]
            #getting selected job key from given function
            if m_function == 'FIFO':
                selected_job = FIFO(poss_jobs, poss_jobs_df)
            if m_function == 'SJF':
                selected_job = SJF(poss_jobs, poss_jobs_df)
            if m_function == 'Random':
                selected_job = Random(poss_jobs, poss_jobs_df)
            if m_function == 'Packer':
                selected_job = Packer(poss_jobs, poss_jobs_df, res_available, res_num)
            if m_function == 'Tetris':
                selected_job = Tetris(poss_jobs, poss_jobs_df, res_available, res_num)
            #checking if job is valid, getting next valid timeslot for job - this is how the packer and tetris move to next T
            if selected_job == 'Null':
                time_pointer = time_pointer + 1
                jobs_left.append(-len(poss_jobs))
                continue
            res_list = poss_jobs.get(selected_job)[0]
            time_pointer = find_valid_loc(res_list, states, time_pointer, res_num, res_cap)
            #marking down the number of jobs left for job_completion valuation
            if time_pointer != last_time_pointer: #if t has changed, only happens with FIFO, SJF, Random b/c of 'Null' in other two
                n_to_write = time_pointer - last_time_pointer
                while(n_to_write >0):
                    jobs_left.append(-len(poss_jobs))
                    n_to_write = n_to_write -1
            #writing job info to all necessary cells
            for i in [*range(0,res_num,1)]:
                for j in [*range(time_pointer, int(time_pointer+res_list[-1]),1)]:
                    states[j, i] = states[j, i] + res_list[i]
            #marking down when job was scheduled to new jobs array (for slowdown)
            job_slowdown[selected_job] = time_pointer-1
            #deleting job key from possible jobs
            del poss_jobs[selected_job]
            poss_jobs_df = poss_jobs_df[poss_jobs_df['i'] != selected_job]

        #writing slowdown info to jobs_df
        slowdown_df = pd.DataFrame.from_dict(job_slowdown, orient='index', columns=[m_function+'_slowdown'] )
        slowdown_df['i'] = job_slowdown.keys()
        jobs_df = jobs_df.merge(slowdown_df, how='outer', on=['i'])

        #writing completion time reward for each timestep
        completion_dict[m_function] = sum(jobs_left)/len(jobs_left)

    return(jobs_df, completion_dict)


"""
Helper functions
"""

#unpacks dict of lists into a single dataframe
def unpack_dict(jobs_dict, n_res):
    jobs_df = pd.DataFrame.from_dict(jobs_dict, orient='index', columns=['list'])
    for i in [*range(0,n_res,1)]:
        jobs_df['res'+str(i)] = jobs_df.apply(lambda row : row['list'][i], axis=1)
    jobs_df['i'] = jobs_dict.keys()
    jobs_df['n_steps'] = jobs_df.apply(lambda row : row['list'][-1], axis=1)
    jobs_df = jobs_df.drop(columns=['list'])
    return(jobs_df)

#creates blank time array
def time_array(length_t, n_resources):
    length = int(length_t*2)
    return (np.zeros(shape=(length, n_resources), dtype=float))

#recursive function to find next valid time slot for job
def find_valid_loc(job_res_list, t_array, t_cur, n_res, cap):
    t_pointer = t_cur
    check = True
    for i in [*range(0,n_res,1)]:
        if job_res_list[i] + t_array[t_pointer, i] > cap:
            check = False
    if check == False:
        t_pointer = find_valid_loc(job_res_list, t_array, t_pointer+1, n_res, cap)
    return(t_pointer)

#tetris function based on n_steps needed and packer score columns
def get_tetris(row, tuner):
    tmp_packer_score = row['packer_score']
    tmp_step_score = 1/row['n_steps']
    return(tuner * tmp_packer_score + (1 - tuner) * tmp_step_score)

"""
Comparison model functions
"""

#First in, First out
def FIFO(poss_jobs_dict, jobs_df):
    return(next(iter(poss_jobs_dict)))

#Selects shortest Job
def SJF(poss_jobs_dict, jobs_df):
    job_i = jobs_df.loc[jobs_df['n_steps'].idxmin()]['i']
    return(job_i)

#Selects Random Job
def Random(poss_jobs_dict, jobs_df):
    poss_keys = [*poss_jobs_dict]
    return(rand.choice(poss_keys))

#Selects job that will best fit remaining available space
def Packer(poss_jobs_dict, jobs_df, res_available, n_res):
    packer_jobs_df = jobs_df
    res_list = []
    #first filters out jobs that won't work
    for i in [*range(0,n_res,1)]:
        packer_jobs_df = packer_jobs_df[packer_jobs_df['res'+str(i)] <= 1-res_available[i]]
        res_list.append('res'+str(i))
    #returns null if no jobs can fit
    if packer_jobs_df.size == 0:
        return("Null")
    #otherwise, returns job with the maximum resources needed combined
    else:
        packer_jobs_df['packer_score'] = packer_jobs_df.loc[:,res_list].sum(axis=1)
        job_i = packer_jobs_df.loc[packer_jobs_df['packer_score'].idxmax()]['i']
        return(job_i)

#Combination of SJF and Packer. Tuner controls whether SJF or Packer favored - 0 to SJF, 1 to Packer
def Tetris(poss_jobs_dict, jobs_df, res_available, n_res, tuner = .5):
    packer_jobs_df = jobs_df
    res_list = []
    #first filters out jobs that won't work
    for i in [*range(0,n_res,1)]:
        packer_jobs_df = packer_jobs_df[packer_jobs_df['res'+str(i)] <= 1-res_available[i]]
        res_list.append('res'+str(i))
    #returns null if no jobs can fit
    if packer_jobs_df.size == 0:
        return("Null")
    #otherwise, creates packer score as designed above, augmented by the shortest possible job
    else:
        packer_jobs_df['packer_score'] = packer_jobs_df.loc[:,res_list].sum(axis=1)
        packer_jobs_df['tetris_score'] = packer_jobs_df.apply(lambda row : get_tetris(row,tuner), axis=1)
        job_i = packer_jobs_df.loc[packer_jobs_df['tetris_score'].idxmax()]['i']
        return(job_i)
from environment import ClusterEnv as ce 

env = ce(70)

grabber = jg(.20, ['CPU', 'GPU'])
jobs, log = grabber.getJobs(70)

env.jobs_profile = jobs
print(compare_models(env.jobs_profile, 2, 1))