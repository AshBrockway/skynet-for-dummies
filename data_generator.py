"""
The purpose of this file is to create a function-level practice for simulating our cluster data as inspired by DeepRM.

Each row will be filled by calling the proposed function. This row will look like:

[(cpu, memory), (duration), which resource is dominant, whether the duration length is long or short]
"""


"""
To do
1. Summarize code into a function that can be called in iterative generation of the data (or change the current code to improve efficiency)
2. Create sample of 100 jobs
    a. the biggest problem to address is to balance the 80% 20% ratio for short and long jobs
    b. consider the data types of the rows themselves as well as how the finished data frame can be created for each sample of job rows
"""
import random
import numpy as np

random.seed(425)

# list of resources and the two classes of job duration (these lists are used to condition and figure our what range the values will be sampled from)
resource_list = ["cpu", "memory"]
dur_list = ["long", "short"]

# chosing the resource and duration length
## note that the overall all data sample should contain 80% short and 20% long, so the duration length is included in the row
primary_resource = random.choice(resource_list)
dur_length = random.choice(dur_list)

# random uniform sample of the resource vec
if primary_resource=="cpu":
    resource_vec = np.random.uniform(low=.25, high=.5, size=1), np.random.uniform(low=.05, high=.1, size=1)
else:
    resource_vec = np.random.uniform(low=.05, high=.1, size=1), np.random.uniform(low=.25, high=.5, size=1)

# random uniform sample of duration length for each job
if dur_length=="long":
    job_duration = np.random.uniform(low=10, high = 15, size = 1)
else:
    job_duration = np.random.uniform(low=1, high=3, size = 1)

# example of finished row
example_row = [list(resource_vec), job_duration, primary_resource, dur_length]

# show example
print(example_row)
