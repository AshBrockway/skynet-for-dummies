# Skynet For Dummies

This repository contains the code from our Practical Data Science project, based off of the [DeepRM](https://people.csail.mit.edu/alizadeh/papers/deeprm-hotnets16.pdf) project.  When run, the program will train a DPN to schedule computer tasks using a Reinforcement Learning algorithm.


### Prerequisites:

pytorch

numpy

pandas

matplotlib

psycopg2 (for database)

grafana (for dashboard)


## Usage:

Run the test.ipynb file after ensuring the prerequisite packages have been installed on your machine.  Must put test file in same directory as the weights.  For optimal performance, run on a machine using a GPU (otherwise it will take a long time)


## Explanation of files:

compare.py - code for the comparison heuristics.

DBconnect.py - code used to export model information to the database (only for use in the dashboard)  Note: to use, you must adjust code with connection parameters for your machine.

enviornment.py - code simulating and updating environment.  Connected with DPN, this consists of the training loop.

execute.py - file used to run entire model, developed for when an optimal policy is trained to, not fully functioning.

grafana - a series of commands to run within a Grafana dashboard (no use directly within this repository)

Test.ipynb - main file used for execution, also produces test plots. NOTE: when you preview this file, it includes the last saved output it ran with.  Also be sure to run this using a GPU connection, as noted in the 'Usage' section

job.py - code to randomly generate jobs and jobsets based on parameters within parameters.py.

Outline_of_DPN_training.py - code containing DPN information for both ANN and CNN models.

parameters.py - code contains model parameters as well as fills initial grid for the enviornment.
