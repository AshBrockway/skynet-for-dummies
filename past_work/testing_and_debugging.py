from parameters import TuneMe as tune
from job import JobGrabber as jg
import numpy as np

tuner = tune()

empty_gridobj = tuner.getGrid()

jobss = jg(.2, ['cpu', 'gpu'])
jobs_set, jobs_log = jobss.getJobs(10)


gilled, back = tuner.fill(jobs_set, empty_gridobj)

#print(gilled)
