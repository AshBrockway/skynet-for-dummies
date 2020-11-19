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

class compare_models:
    def __init__(self, job_dict, res_num, res_cap):
        #pulling in necessary parameters
        n_jobs = len(job_dict)
        model_functions = ['FIFO','SJF','Random','Packer']#,'Tetris']

        #transforming job dictionary into dataframe
        jobs_df = unpack_dict(job_dict, n_res=res_num)

        #transforms df to include slowdown information for every job on the dataframe
        jobs_df = process_df(jobs_df, job_dict, model_functions, res_num, n_jobs, res_cap)
        self.full_jobs_df = jobs_df

        #slowdown variables for later storage
        self.FIFO_loss = jobs_df['FIFO_slowdown'].mean()
        self.SJF_loss = jobs_df['SJF_slowdown'].mean()
        self.Random_loss = jobs_df['Random_slowdown'].mean()
        self.Packer_loss = jobs_df['Packer_slowdown'].mean()
        # self.Tetris_loss =


"""
Primary function
"""
#processes jobs_df to include comparison models
def process_df(jobs_df, job_dict, model_functions, res_num, n_jobs, res_cap):
    for m_function in model_functions:
        #initializing time array to track jobs/slowdown, other inits
        states = time_array(n_jobs*2, res_num)
        time_pointer = 1
        job_slowdown = {}
        #copying job dict for use in loop
        poss_jobs = job_dict.copy()
        poss_jobs_df = jobs_df
        #looping through to assign jobs based on the model
        while len(poss_jobs) > 0:
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
                selected_job = Tetris(poss_jobs, poss_jobs_df, res_available)
            #checking if job is valid, getting next valid timeslot for job
            if selected_job == 'Null':
                time_pointer = time_pointer + 1
                continue
            res_list = poss_jobs.get(selected_job)[0]
            time_pointer = find_valid_loc(res_list, states, time_pointer, res_num, res_cap)
            #writing job info to all necessary cells
            for i in [*range(0,res_num,1)]:
                for j in [*range(time_pointer, int(time_pointer+res_list[-1]),1)]:
                    states[j, i] = states[j, i] + res_list[i]
            #marking down when job was scheduled to new jobs array
            job_slowdown[selected_job] = time_pointer-1
            #deleting job key from possible jobs
            del poss_jobs[selected_job]
            poss_jobs_df = poss_jobs_df[poss_jobs_df['i'] != selected_job]

        #writing slowdown info to jobs_df
        slowdown_df = pd.DataFrame.from_dict(job_slowdown, orient='index', columns=[m_function+'_slowdown'] )
        slowdown_df['i'] = job_slowdown.keys()
        jobs_df = jobs_df.merge(slowdown_df, how='outer', on=['i'])
    return(jobs_df)


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

"""
Comparison model functions
"""

def FIFO(poss_jobs_dict, jobs_df):
    return(next(iter(poss_jobs_dict)))


def SJF(poss_jobs_dict, jobs_df):
    job_i = jobs_df.loc[jobs_df['n_steps'].idxmin()]['i']
    return(job_i)

def Random(poss_jobs_dict, jobs_df):
    poss_keys = [*poss_jobs_dict]
    return(rand.choice(poss_keys))

def Packer(poss_jobs_dict, jobs_df, res_available, n_res):
    packer_jobs_df = jobs_df
    res_list = []
    for i in [*range(0,n_res,1)]:
        packer_jobs_df = packer_jobs_df[packer_jobs_df['res'+str(i)] <= 1-res_available[i]]
        res_list.append('res'+str(i))
    if packer_jobs_df.size == 0:
        return("Null")
    else:
        packer_jobs_df['packer_score'] = packer_jobs_df.loc[:,res_list].sum(axis=1)
        job_i = packer_jobs_df.loc[packer_jobs_df['packer_score'].idxmax()]['i']
        return(job_i)

def Tetris(poss_jobs_dict, jobs_df):

    """
    Tetris function (tbd)
    """
    pass


