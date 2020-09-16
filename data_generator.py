"""
The purpose of this file is to create a function-level practice for simulating our cluster data as inspired by DeepRM.

Each row will be filled by calling the proposed function. This row will look like:

[(cpu, memory), (duration), which resource is dominant, whether the duration length is long or short]
"""


"""
To do
1. Summarize code into a function that can be called in iterative generation of the data (or change the current code to improve efficiency) DONE
2. Create sample of 100 jobs
    a. the biggest problem to address is to balance the 80% 20% ratio for short and long jobs DONE
    b. consider the data types of the rows themselves as well as how the finished data frame can be created for each sample of job rows DONE
"""
import random
import numpy as np

def data_gen(n_rows):
    random.seed(425)
    countShort = 0
    countLong = 0
    twentySplit = int(n_rows*.20)
    eightySplit = int(n_rows*.80)
    for i in range(1, n_rows+1):
        # list of resources and the two classes of job duration (these lists are used to condition and figure our what range the values will be sampled from)
        resource_list = ["cpu", "memory"]
        dur_list = ["long", "short"]

        # choosing the resource and duration length
        ## note that the overall data sample should contain 80% short and 20% long, so the duration length is included in the row
        primary_resource = random.choice(resource_list)
        dur_length = random.choice(dur_list)
        print(dur_length)
        #inventive way to ensure a 20/80 split of long and short jobs
        if dur_length == "short":
            countShort += 1
        if dur_length == "long":
            countLong += 1
        if countShort > eightySplit and countLong < twentySplit:
            dur_length == "long"
            countShort -= 1
            countLong += 1
        if countShort < eightySplit and countLong > twentySplit:
            dur_length = "short"
            countShort += 1
            countLong -= 1
        #print(dur_length)

        # random uniform sample of the resource vec
        if primary_resource=="cpu":
            resource_vec = float(np.random.uniform(low=.25, high=.5, size=1)), float(np.random.uniform(low=.05, high=.1, size=1))
        else:
            resource_vec = float(np.random.uniform(low=.05, high=.1, size=1)), float(np.random.uniform(low=.25, high=.5, size=1))

        # random uniform sample of duration length for each job
        if dur_length=="long":
            job_duration = float(np.random.uniform(low=10, high = 15, size = 1))
        else:
            job_duration = float(np.random.uniform(low=1, high=3, size = 1))

        # example of finished row
        example_row = [list(resource_vec), job_duration, primary_resource, dur_length]

        # show example
        print(example_row)
    data_info = [n_rows,twentySplit, countLong, eightySplit, countShort]
    print(data_info)
data_gen(10)

