"""
Author: Ashley Brockway, ashley.brockway15@ncf.edu
Date: 9/30/20
Purpose: ....

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

import paramaters.Tune_Me as pa
import job as js


class ClusterEnv:
    def __init__(self, set_length=10):
        """
         Pull from 2 other files :
            1. the parameters of our environment
                a. These create a class based on all the stuff we want to be available
                to easily change and access
            2. jobs that are entering our environment
        """

        # Build grid using dimensions from paramater.py
        # # # # TODO (@ash) create a default parameter for the constructor to automatically build given the parameter is set to T its default is F

        # Construction of empty state
        self.obs_state = pa(build_on_creation=True)

        # Grab some jobs
        self.jobs = js(set_length)

        # Populate the empty state with the jobs
        self.obs_state.fill(self.jobs)
