"""
Author: Ashley Brockway, ashley.brockway15@ncf.edu
Date: 9/30/20
Purpose:
The purpose of this file is to generate any given number of jobs in order to fill the empy grid
space. This filled grid will then be used as input into the Deep Policy Network.

Description:
The purpose is to take an input of the length of a jobset to be constructed for
1 episode of training. One job will consist of both a time duration and a resource vector.
The dimensions of the resource vector are stored in parameters.py in the var. res_num.

For DeepRM the resource vector will have 2 elements and look like this:
    (r_1, r_2)
    Where r_1 is % of CPU est. necessary for job, and r_2 is % of GPU est..
But wait, you are probably wondering how we will get values of r_1 and r_2, there are a few layers to this:
    T

Meaning a set of 3 jobs will look like this:

    Job 1:  ((r_{1,1} , r_{1, 2}), T_{1})
    Job 2:  ((r_{2,1} , r_{2, 2}), T_{1})
    Job 3:  ((r_{3,1} , r_{3, 2}), T_{1})

"""



class Jobset:
    """
    The parameters in the constructor for this class are:
        1. set_num = the
    """
    def __init__(self, lt_prop, resource_list):

        # time duration
        self.long_time = lt_prop # distribution propoerty of "long jobs"
        self.short_time = 1 - lt_prop

        # get list of resource names
        self.res_list = reousrce_list

    def getJobs(self, set_num):
        self.dom_res = rand.sample(self.res_list, 1)
        for i in 1:set_num:
            pass


# define job class
# # # see description at top o'page to visualize what a job consists of

class Job:
    """
    long: True or False for whether this is a long or a short jobs
    dom_res: a string representing the dominant resource
    """
    def __init__(self, long, dom_res):
        # Get resource vector
        # # Find the dominant resource by
        pass
