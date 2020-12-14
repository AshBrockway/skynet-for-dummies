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

Run the execute.py file after ensuring the prerequisite packages have been installed on your machine.


## Explanation of files:

compare.py - code for the comparison heuristics

DBconnect.py - code used to export model information to the database (only for use in the dashboard)

enviornment.py - code simulating and updating environment.  Connected with DPN, this consists of the training loop.

execute.py - file used to run entire model

job.py - code to randomly generate jobs and jobsets based on parameters within parameters.py

Outline_of_DPN_training.py - code containing DPN information for both ANN and CNN models

parameters.py - code contains model parameters as well as fills initial grid for the enviornment
