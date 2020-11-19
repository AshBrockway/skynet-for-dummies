# Skynet For Dummies

This repo is for our Practical Data Science project, initially recreating the [DeepRM](https://people.csail.mit.edu/alizadeh/papers/deeprm-hotnets16.pdf) project
before moving on to different applications of Deep Reinforcement Learning.


### Prerequisites:

pytorch

numpy

matplotlib

psycopg2

grafana


## Usage:

(To be completed - something about running execute.py file when it is in)

## Big TODO list:

Note - If a job has been completed, <s>add a strikethrough to that line</s> using "s and /s" syntax in README file

#### Create execute.py file
Ya know, just the master execute file that will pull all of these other classes together to run everything.


#### Create export.py file
This file will take parameters from other classes and export them to a postgres database


#### In parameters.py file:

1. TODO - TuneMe.fill() function - (@ash)

    Explanation - create iterative method to place values in Grid elements keeping padding in mind


#### In enviornment.py file:

1. TODO - Class Enviorn - (UNCLAIMED)

    Explanation - ?

2. TODO - Create default parameter (under ClusterEnv.__init__()) - (UNCLAIMED)

    Explanation - create a default parameter for the constructor to automatically build given the parameter is set to T its default is F

3. TODO - ClusterEnv.updateState() function - (@ash)

    Explanation - given that a job has been chosed, move its resource usage into the relevant grids


#### In job.py file:

1. TODO - JobGrabber.__str__() - (UNCLAIMED)

    Explanation - this is the string method to define what we will see when we want to print job class



#### In data_generator.py file:

(none currently)


#### In tuning.py file:

(none currently)


#### In Outline_of_DPN_training.py file:

1. TODO - Define the policy network for training - ()

      Explanation - Define the model that we'll be training. Worth experimenting with architecture, and so forth.


2. <s>TODO - curried_valuation.valuation()</s> - (@justin)


    <s>Explanation - compute valuation with the episode/trajectory after it's been padded. Result intended to be used as
    map(valuation, episodes_array) to return valuation of each episodes.</s>

3. <s>TODO - DPN.train_on_jobs()</s> - (@justin)

    <s>Explanation - figured out how to access the DPN's gradient shape and initialize it to all zeros </s>

4. <s>TODO - DPN.train_on_jobs()</s> - (@justin)

    <s>Explanation - Figured out how to update the gradient sum based on the policy chosen with the relativized reward. Moreover, figured out how to update the weights of the network after summing the relativized gradient updates. </s>



### Long-term task list (Not necessary for basic model as of 10/4/2020)

1. Convert data_generator.py file into proper class.function() form

2. Adapt enviornment code to update using custom openai gym for easy generalizing

3. Convert 2D numpy environment into 3D numpy environment to simplify padding issues

4. ... ?
