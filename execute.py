"""
Author: Ashley Brockway (ashley.brockway15@ncf.edu), Andrew Reilly (andrew.reilly19@ncf.edu)
Date: 10/10/20
Purpose: This file will execute the model, calling each individual component's class and running each at the correct time.

"""

#package imports
import numpy as np
import pandas as pd

#Other file imports
from parameters import TuneMe as pa
#from tuning import TuneMe as TuneMe2
from job import JobGrabber
from DBconnect import DBconnect as DB
#from environment import ClusterEnv as Env
#from Outline_of_DPN_training import DPN
from compare import compare_models as comp



"""
Steps:
    1. set model parameters using 'TuneMe' class in parameters.py and tuning.py
        1b. Eventually - passing in parameters from Dashboard
    2. create jobs for model to run with 'JobGrabber' class in job.py
    3. export initial parameters and jobs to database with DBConnect
    4. pass parameters and jobs to environment for construction
    5. repeat next steps:
        5a. Pass enviornment to DPN training for policy analysis/selection  (If using alternate method, use that here instead)
        5b. Pass policies back to environment, process and create next stage
        5c. Update datbase with what jobs were completed and the stage
        ...
        5a. Pass enviornment to DPN training for policy analysis/selection
        ...
    Repeat until all jobs have been completed
    6. process final stats for model - total slowdown or total time to completion
    7. output final stats to database

Further considerations:
    1. Do we want the alterantive models to run concurrently with DPN model, or separately? If concurrently, will impact performance
        and if separately, we will need some way to save jobs to use again.
    2. Comparison models will not take the image that the DPN will use, so we will need to figure out how to directly import jobs/env to these models

TODO: set up code for this file
"""


#getting/setting parameters, etc.
pars = pa()
res_num = pars.res_num
res_cap = pars.res_cap


#getting jobs
n_jobs = 30
job_obj = JobGrabber(.2, ["cpu", "gpu"])
job_data, job_info = job_obj.getJobs(n_jobs)

#comparison models object creation
comp_models = comp(job_data, res_num, res_cap)

df = comp_models.full_jobs_df
print(comp_models.FIFO_loss)
print(comp_models.SJF_loss)
print(comp_models.Random_loss)
print(comp_models.Packer_loss)




