"""
Author: Ashley Brockway (ashley.brockway15@ncf.edu), Andrew Reilly (andrew.reilly19@ncf.edu)
Date: 10/10/20
Purpose: This file will execute the model, calling each individual component's class and running each at the correct
time.

"""


from parameters import TuneMe as Pa
from job import JobGrabber
from DBconnect import DBconnect
from DBconnect import get_last_iter
from environment import ClusterEnv as Env
from Outline_of_DPN_training import DPN, DP_CNN
from compare import compare_models as comp



from Outline_of_DPN_training import DPN
from environment import ClusterEnv

intera_num = 1000

start = ClusterEnv(70)
dpn = DPN(start)

dpn.train(itera_num)


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


DBconnect LIST FORMATS:

#VALUES must be in list form: [n_resources, syscap, %long, %short, resourcelow, resourcehigh, timelow, timehigh, n_jobs, n_epochs]
#OVERALL LOSSES must be in list form: [DPN_loss, DPCNN_loss, AC_loss, FIFO_loss, SJF_loss, Random_loss, Packer_loss, Tetris_loss]
INTERMEDIATE LOSSES must be in list form: [DPN_loss, DPCNN_loss, AC_loss, FIFO_loss, SJF_loss, Random_loss, Packer_loss, Tetris_loss]

"""

"""

def main():
    #set new parameters here
    pa = Pa()

    #getting last iteration from database so we know what we're on
    last_iteration = get_last_iter()
    iteration = last_iteration + 1

    #updating database with parameters
    #FORM: [n_resources, syscap, %long, %short, resourcelow, resourcehigh, timelow, timehigh, n_jobs, n_epochs]
    param_list = [pa.res_num, pa.res_cap, pa.long, 1-pa.long, pa.reslow, pa.reshigh, pa.timelow, pa.timehigh, pa.jobs, pa.epochs]
    db = DBconnect(iteration)
    db.update_params(param_list)

    #create res list
    if pa.res_num == 2:
        resource_names = ['cpu','gpu']
    else:
        resource_names = []
        for i in [*range(1,pa.res_num+1, 1)]:
            resource_names.append('res_'+str(i))
    job_obj = JobGrabber(pa.long, resource_names)

    jobset_list = {}
    for i in [*range(1, pa.epochs+1,1)]:
        pass


"""


'''
#getting/setting parameters, etc.
pars = pa()
res_num = pars.res_num
res_cap = pars.res_cap


#getting jobs
job_obj = JobGrabber(.2, ["cpu", "gpu"])
job_data, job_info = job_obj.getJobs(n_jobs)

#comparison models object creation
comp_models = comp(job_data, res_num, res_cap)

#df needed only to double check things
#df = comp_models.full_jobs_df
print(comp_models.FIFO_loss)
print(comp_models.SJF_loss)
print(comp_models.Random_loss)
print(comp_models.Packer_loss)
print(comp_models.Tetris_loss)


#Running the DPN
# emptyGrid = pars.getGrid()
# ggrid = pars.fill(job_data, emptyGrid)


# ggrid2 = ggrid.flatten()
# len(ggrid2)
'''
