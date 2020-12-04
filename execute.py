"""
Author: Ashley Brockway (ashley.brockway15@ncf.edu), Andrew Reilly (andrew.reilly19@ncf.edu)
Date: 10/10/20
Purpose: This file will execute the model, calling each individual component's class and running each at the correct
time.

"""


from parameters import TuneMe as Pa
from job import JobGrabber
#from DBconnect import DBconnect
#from DBconnect import get_last_iter
from environment import ClusterEnv as Env
from Outline_of_DPN_training import DPN, DP_CNN
from compare import compare_models as comp



"""
Steps:
    1. set model parameters using 'TuneMe' class in parameters.py and tuning.py
        1b. Eventually - passing in parameters from Dashboard
    2. create jobs for model to run with 'JobGrabber' class in job.py
    3. export initial parameters and jobs to database with DBConnect
    4. pass parameters and jobs to environment for construction
    5. repeat next steps:
        5a. Pass enviornment to DPN training for policy analysis/selection  (If using alternate method, use that here
        instead)
        5b. Pass policies back to environment, process and create next stage
        5c. Update database with what jobs were completed and the stage
        ...
        5a. Pass environment to DPN training for policy analysis/selection
        ...
    Repeat until all jobs have been completed
    6. process final stats for model - total slowdown or total time to completion
    7. output final stats to database

Further considerations:
    1. Do we want the alternative models to run concurrently with DPN model, or separately? If concurrently, will impact
    performance.
        and if separately, we will need some way to save jobs to use again.
    2. Comparison models will not take the image that the DPN will use, so we will need to figure out how to directly import jobs/env to these models


DBconnect LIST FORMATS:

#VALUES must be in list form: [n_resources, syscap, %long, %short, resourcelow, resourcehigh, timelow, timehigh, n_jobs, n_epochs]
#OVERALL LOSSES must be in list form: [DPN_loss, DPCNN_loss, AC_loss, FIFO_loss, SJF_loss, Random_loss, Packer_loss, Tetris_loss]
INTERMEDIATE LOSSES must be in list form: [DPN_loss, DPCNN_loss, AC_loss, FIFO_loss, SJF_loss, Random_loss, Packer_loss, Tetris_loss]

"""



# def main():
#     #set new parameters here
#     pa = Pa()

#     #getting last iteration from database so we know what we're on
#     last_iteration = get_last_iter()
#     iteration = last_iteration + 1

#     #updating database with parameters
#     #FORM: [n_resources, syscap, %long, %short, resourcelow, resourcehigh, timelow, timehigh, n_jobs, n_epochs]
#     param_list = [pa.res_num, pa.res_cap, pa.long, 1-pa.long, pa.reslow, pa.reshigh, pa.timelow, pa.timehigh, pa.jobs, pa.epochs]
#     db = DBconnect(iteration)
#     db.update_params(param_list)

#     #create res list
#     if pa.res_num == 2:
#         resource_names = ['cpu','gpu']
#     else:
#         resource_names = []
#         for i in [*range(1,pa.res_num+1, 1)]:
#             resource_names.append('res_'+str(i))
#     job_obj = JobGrabber(pa.long, resource_names)

#     #creating big list of cluster enviornments (jobsets) to iterate over)
#     CE_list = []
#     for i in [*range(1, pa.jobsets+1,1)]:
#         c_obj = Env(pa.jobs)
#         CE_list.append(c_obj)

#     #begin training iterations
#     for i in [*range(1, pa.epochs+1,1)]:
#         job_data, job_info = job_obj.getJobs(pa.jobs)







#getting/setting parameters, etc.
pars = Pa()
res_num = pars.res_num
res_cap = pars.res_cap
n_jobs = pars.jobs

#getting jobs
job_obj = JobGrabber(.2, ["cpu", "gpu"])

#initializing comp models completion time lists
FIFO_CT = []
SJF_CT = []
Random_CT = []
Packer_CT = []
Tetris_CT = []

FIFO_slow = []
SJF_slow = []
Random_slow = []
Packer_slow = []
Tetris_slow = []

comp_iterations = 100
while comp_iterations >0:
    job_data, job_info = job_obj.getJobs(n_jobs)

    #comparison models object creation
    comp_models = comp(job_data, res_num, res_cap)

    FIFO_CT.append(comp_models.FIFO_actime)
    SJF_CT.append(comp_models.SJF_actime)
    Random_CT.append(comp_models.Random_actime)
    Packer_CT.append(comp_models.Packer_actime)
    Tetris_CT.append(comp_models.Tetris_actime)

    FIFO_slow.append(comp_models.FIFO_loss)
    SJF_slow.append(comp_models.SJF_loss)
    Random_slow.append(comp_models.Random_loss)
    Packer_slow.append(comp_models.Packer_loss)
    Tetris_slow.append(comp_models.Tetris_loss)

    comp_iterations -= 1


FIFO_CT_fin = sum(FIFO_CT)/len(FIFO_CT)
SJF_CT_fin = sum(SJF_CT)/len(SJF_CT)
Random_CT_fin = sum(Random_CT)/len(Random_CT)
Packer_CT_fin = sum(Packer_CT)/len(Packer_CT)
Tetris_CT_fin = sum(Tetris_CT)/len(Tetris_CT)

print(FIFO_CT_fin)
print(SJF_CT_fin)
print(Random_CT_fin)
print(Packer_CT_fin)
print(Tetris_CT_fin)


FIFO_slow_fin = sum(FIFO_slow)/len(FIFO_slow)
SJF_slow_fin = sum(SJF_slow)/len(SJF_slow)
Random_slow_fin = sum(Random_slow)/len(Random_slow)
Packer_slow_fin = sum(Packer_slow)/len(Packer_slow)
Tetris_slow_fin = sum(Tetris_slow)/len(Tetris_slow)
print("\n")
print(FIFO_slow_fin/n_jobs)
print(SJF_slow_fin/n_jobs)
print(Random_slow_fin/n_jobs)
print(Packer_slow_fin/n_jobs)
print(Tetris_slow_fin/n_jobs)



#df needed only to double check things
#df = comp_models.full_jobs_df



#Running the DPN
# emptyGrid = pars.getGrid()
# ggrid = pars.fill(job_data, emptyGrid)


# ggrid2 = ggrid.flatten()
# len(ggrid2)


