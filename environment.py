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

TODO
Class Environ
"""

from parameters import TuneMe as pa
from job import JobGrabber as jg


class ClusterEnv:
    def __init__(self, set_length):
        """
         Pull from 2 other files :
            1. the parameters of our environment
                a. These create a class based on all the stuff we want to be available
                to easily change and access
            2. jobs that are entering our environment
        """

        # Build grid using dimensions from paramater.py
        # # # # TODO create a default parameter for the constructor to automatically build given the parameter is set to T its default is F

        # Construction of empty state
        objs_state = pa()
        self.obs_state = objs_state.getGrid()

        # percent of long jobs
        pct_lj = .2

        # Grab some jobs
        jobs = jg(pct_lj, ["cpu", "gpu"])
        # jobs_profile is the values themselfs, log is the string info about the jobs
        self.jobs_profile, self.jobs_log = jobs.getJobs(set_length)

        # Populate the empty state with the jobs
        self.filled, self.backlog = objs_state.fill(self.jobs_profile, self.obs_state)

    # TODO (@ash)
    # # given that a job has been chosed, move its resource usage into the relevant grids
    # # # if time_step=T then the rows will also shift up so that T_{0+1} becomes the top row and a new last shown timestep is included
    def updateState(self, job_choice, time_step=True):
        pass

env = ClusterEnv(10)



print(env.filled)
