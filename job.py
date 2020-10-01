"""
Author: Ashley Brockway, ashley.brockway15@ncf.edu
Date: 9/30/20
Purpose:
The purpose of this file is to produce a job set with a specified number of jobs.

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

Because of the job method including that the set be 20% long time duration and 80% short time duration, and randomly choose which resource would be domination (dominant: [0.25, .50], non-dominant: [.05, .10]) a log will be kept.

"""

import numpy as np
import random as rand


class JobGrabber:
    """
    The parameters in the constructor for this class are:
        lt_prop: the proportion of long time durations
        resource_list: a list of the string names of the resources
    The JobGrabber class utlizes the Job class within its getJobs method to combine a set of job's
    resource and time profiles.

    jobs is the resource and time profiles of the jobset.
    And jobs_info is the job log with information on the ratio and dominant resource of a given job.


    """
    def __init__(self, lt_prop, resource_list):

        # time duration ratio
        self.long_time = lt_prop # proportion of "long jobs"
        self.short_time = 1 - lt_prop # proportion of "short jobs"


        # get list of resource names
        self.res_list = resource_list

    # Method to get a job of set_num length, meaning set_num=#of jobs in set
    def getJobs(self, set_num):
        # Create lists of 0 that match the specified number of jobs for job profiles and logs
        jobs = [ 0 for x in range(set_num) ]
        jobs_info = [ 0 for x in range(set_num) ]

        # For set_num iterations:
        for i in range(set_num):
            # randomly choose one of the resources to be dominant
            dom_res = rand.sample(self.res_list, 1)
            # on the condition that we are grabbing the proportion of short jobs
            if i <= ((set_num * self.short_time) - 1):
                # Calls an object of the Job class defined below, as a short job with the given dom. res
                job_ob = Job(False, dom_res, self.res_list)
            # then for any long jobs
            else:
                job_ob = Job(True, dom_res, self.res_list)

            # then for each job fill the jobs and the job_info list
            jobs[i] = job_ob.job_info
            jobs_info[i] = job_ob.job_data

        # shuffle the jobs so that their order isnt defined off their time duration
        rand.shuffle(jobs)

        # return the jobs and job_info lists made up of the profiles and the log info respectively
        return(jobs, jobs_info)

    # this is the string method to define what we will see when we want to print job class
    # # TODO (@ash) understand how this works
    # def __str__(self):
        # pass



class Job:
    """
    loonggg: True or False for whether this is a long or a short jobs
    dom_res: a string representing the dominant resource

    This code is adapted from the data_generator.py by @JamieHobbs
    """
    # Construction of a Job
    def __init__(self, loonggg, dom_res, res_list):
        # list of 0s of the length of the resource vector (i.e. how many resources we have)
        res_vec = [0 for x in range(len(res_list))]
        # this is an identical list for the log information for the job
        self.job_data = [0 for x in range(len(res_list))]
        # enumerating the resource list, we check for whether the resource was chosen as dominant
        # # and then sample that resources usage
        for index, res in enumerate(res_list):
            if res==dom_res:
                res_vec[index] = float(np.random.uniform(low=.25, high=.5, size=1))
                # concat. strings to make a "this resource is dominant" entry in the log
                self.job_data[index] = res + " is dominant"
            else:
                res_vec[index] = float(np.random.uniform(low=.05, high=.1, size=1))
                self.job_data[index] = res + " is not  dominant"

        # Check the duration label for the given job
        if loonggg==True:
            job_duration = float(np.random.uniform(low=10, high = 15, size = 1))
            # store the label in the log
            self.job_data.append("Long Job")
        else:
            job_duration = float(np.random.uniform(low=1, high=3, size = 1))
            self.job_data.append("Short Job")

        # create the resource vector for the job
        self.job_info = res_vec
        # create the job log for the job
        self.job_info.append(job_duration)



# for testing
# job_0b = JobGrabber(.2, ["cpu", "gpu"])

# jobs, info = job_0b.getJobs(10)