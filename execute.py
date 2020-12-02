"""
Author: Ashley Brockway (ashley.brockway15@ncf.edu), Andrew Reilly (andrew.reilly19@ncf.edu)
Date: 10/10/20
Purpose: This file will execute the model, calling each individual component's class and running each at the correct
time.

"""

from parameters import TuneMe as pa
# from job import JobGrabber as jg
# from DBconnect import DBconnect as db
# from environment import ClusterEnv as Env
# from Outline_of_DPN_training import DPN
# from compare import SJF, Packer, FIFO, Tetris


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

TODO: set up code for this file
"""


def main():
    objs_state = pa()
    objs_state.updateDB("params")

main()
