"""
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

# characteristics of our environment


class ClusterEnv:
    def __init__(self, job_set):
        """
         Pull from 2 other files :
            1. the parameters of our environment
                a. These create a class based on all the stuff we want to be available
                to easily change and access
            2. jobs that are entering our environment
        """
