"""
WARNING:

Code will not be executable from this alone. Must clean up the TODOs.

Provides a backbone for the training method on a deep policy network.

Presumably, the input to the network will be images as the current state (including current resource usage, and random M jobs to schedule).
The output predictions are the probabilities for choosing to schedule one of the M jobs.

Most important is the train_on_jobs method for updating the weights of the network, in accordance with DPN loss functions.
"""


import numpy as np
import random
import torch
from torch.distributions import Categorical
from torch import nn, optim
from environment import ClusterEnv as CE 

from parameters import TuneMe as pa


ITERATIONS = 1000 #kinda like epochs?
BATCH_SIZE = 10   #Might be the exact same thing as episodes, up for interpretation.
EPISODES = 20     #How many trajectories to explore for a given job. Essentually to get a better estimate of the expected reward.
DISCOUNT = 0.99   #how much to discount the reward
ALPHA = 0.001     #learning rate?

#TODO: WHAT IS GAMMA??? (used in valuation function, line 56(ish))
gamma = .99

def curried_valuation(length_of_longest_trajectory):
    '''
    Given the length of the longest trajectory of a set of episodes;

    returns the function that will compute the valuation of an episode array (while padding it)

    Result intended to be used as  map(valuation, episodes_array) to return valuation of each episodes.
    '''
    def valuation(episode):
        '''
        returns the valuation of an episode (with padding)
        input: [(s_0, a_0, r_0), ... ,(s_t, a_t, r_t)]         potentially t<L
        output: [v_0, v_1, ... v_L]
        '''

        length = len(episode)
        if length != length_of_longest_trajectory:
            #If the episode isn't as long as the longest trajectory, pad it
            episode.extend([(0,0,0) for y in range(length_of_longest_trajectory-length)]) #have to make sure the numbers line up correctly
        out = np.zeros(len(episode))
        x = [i[2] for i in episode] #rewards
        out[-1] = x[-1]
        for i in reversed(range(len(x)-1)): #go backwards
            out[i] = x[i] + gamma*out[i+1] #this step valuation = reward + gamma*next_step_valuation
        #assert x.ndim >= 1
        return out
    return valuation



class DPN:  #ANN with Pytorch
    def __init__(self, enve):
        self.n_inputs = len(enve.filled.flatten())
        #TODO: Make outputs reflexive
        self.n_outputs = 11
        self.env = enve

        # Define network
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Softmax(32, self.n_outputs))

    def predict(self, state):
        state = state.flatten()
        action_probs = self.network(torch.FloatTensor(state))
        return action_probs


        all caps words are hyperparameters you would set.
        '''


        params = pa()
        # get number of jobs
        self.jobs = params.getNumJobs()
        print("Jobs = " + str(self.jobs))

        input_features = np.array()
        target_output = np.array()
        target_output = target_output.reshape()
        weigths = np.array()

        bias = 0.3
        lr = 0.05

        def sigmoid(x):
            return 1/(1+np.exp(-x))
        def sigmoid_der(x):
            return sigmoid(x)*(1-sigmoid(x))

        for epoch in range(10000):
            inputs = input_features
            pred_in = np.dot(input, weights) + bias
            pred_out = sigmoid(pred_in)
            error = pred_out - target_output
            x = error.sum()

            print(x)

            dcost_dpred = error
            dpred_dx = sigmoid_der(pred_out)

            z_delta = dcost_dpred * dpred_dx

            inputs = inputs_features.T
            weights -= lr * np.dot(inputs, z_delta)

            for i in z_delta:
                    bias -= lr * i

        single_point = n.array()
        result1 = np.dot(single_point, weights) + bias

        result2 = sigmoid (result1)

        print(result2)


        self.prob_history = {} #Might be a dumb idea, but it stores the probability that the network chose the action given the state?

                               # key looks like (state, action) value is the probability? marked with optional1
    def train(self, ITERATIONS):
        '''
        Might need to be moved outside model definition due to how pytorch works????

        Defines optimizer which takes steps to update our model weights. Can start with them all at 0.
        Loops through each iteration, makes a new set of jobs after constructing the resource constraints and time from the environment.
        Then does a training iteration on those jobs.
        '''
        #Let's consider a different optimizer, but this is just proof of concept. When you define a NN in pytorch, the class inherits a parameters method that's supplied to the optimizer. Hence next comment:
        optimizer = optim.Adam(self.network.parameters(), lr=1e-3) #This kind of definition might actually have to be defined outside of our model class. TODO 1/2: split model definition from training functions. Half for readability, half to work???
        for i in range(ITERATIONS):
            
            self.train_on_jobs(optimizer)


    def trajectory(self, current_state_env):
        '''
        Maybe this implementation doesn't utilize GPUs very well, but I have no clue or not.

        Final output looks like:
        [(s_0, a_0, r_0), ..., (s_L, a_L, r_l)]
        '''
        
        output_history = []
        while True:
            probs = self.predict(current_state_env.filled)#could be self.predict()   TODO (by model building, or custom implementation). Basically define model architecture
            picked_action = Categorical(probs).sample() #returns index of the action/job selected.
            #self.prob_history[(current_state, picked_action)] = choice_prob #optional1
            new_state = current_state_env.updateState( picked_action, current_state) #Get the reward and the new state that the action in the environment resulted in. None if action caused death. TODO build in environment
            reward = sum(current_state_env.rewards)
            
            output_history.append( (current_state, picked_action, reward) )
            if new_state is None: #essentially, you died or finished your trajectory
                break
            else:
                current_state = new_state
        return output_history

    def train_on_jobs(self, optimizer):
        '''
        Training from a batch. Kinda presume the batch is a set of starting states not sure how you have the implemented states (do they include actions internally?)

        example shape of episode_array
        [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 6, 7],
        [1, 2, 3]
        ]
        '''
        optimizer.zero_grad() #This sets the optimizer to update the weights by 0. We'll add to it over time! TODO: Check that it actually works
        for job_start in range(101):
            #episode_array is going to be an array of length N containing trajectories [(s_0, a_0, r_0), ..., (s_L, a_L, r_0)]
            self.envi = CE(70)
            episode_array = [self.trajectory(self.envi) for x in range(EPISODES)]
            # Now we need to make the valuations
            longest_trajectory = max(len(episode) for episode in episode_array)
            valuation_fun = curried_valuation(longest_trajectory)
            cum_values = np.array([valuation_fun(ep) for ep in episode_array]) #should be a EPISODESxlength sized
            #can compute baselines without a loop?
            baseline_array = np.array([sum(cum_values[:,i])/EPISODES for i in range(longest_trajectory)]) #Probably defeats the purpose of numpy, but we're essentially trying to sum each valuation array together, and then divide by the number of episodes TODO make it work nicely
            for i in range(EPISODES): #swapped two for loops
                for t in range(longest_trajectory):
                    try:
                        state, action, reward = episode_array[i][t]
                    except IndexError: #this occurs when the trajectory died
                        break
                    #get probabilities from the network. We already did this, but pretty sure we gotta do it again.
                    probs = self.predict(state)
                    DPN_Theta = Categorical(probs) #Pytorch distribution for Categorical classes. SHOULD connect to the network to update weights.
                    if i == 0 and t == 0: #Define the first loss in the sum
                        loss = -(cum_values[i][t]-baseline_array[t])*ALPHA*DPN_Theta.log_prob(action)
                    else: #Keep adding to the loss
                        loss += -(cum_values[i][t]-baseline_array[t])*ALPHA*DPN_Theta.log_prob(action) #This is what it _should_ look like in pytorch. Added negative (trying to maximize reward, but we're trying to find a minimum) on recommendation of pytorch documentation: https://pytorch.org/docs/stable/distributions.html
        loss.backward() #Compute the total cumulated gradient thusfar through our big-ole sum of losses
        optimizer.step() #Actually update our network weights. The connection between loss and optimizer is "behind the scenes", but recall that it's dependent


dpn = DPN(CE(70))

dpn.train(10)
