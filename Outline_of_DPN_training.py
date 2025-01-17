"""
The purpose of this file is to provide training code for DPN with ANN and CNN,
either can be trained after the object is initialized with any environment. 
"""


import numpy as np
import random
import torch
from torch.distributions import Categorical
from torch import nn, optim
from environment import ClusterEnv as CE

from parameters import TuneMe as pa


ITERATIONS = 1000 # number of iterations the model will run for (the number of times train_on_jobs method is used in train)
EPISODES = 20   # How many trajectories to explore for a given job. Helps reduce variance in a reward calculation for a given jobset
ALPHA = 0.01     # learning rate in batch parameter updates
gamma = .99 # discount in cummulative reward

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
        rews = [i[2] for i in episode] #rewards
        out[-1] = rews[-1]
        for i in reversed(range(0, len(rews)-1)):
            #go backwards
            out[i] = rews[i] + gamma*out[i+1] #this step valuation = reward + gamma*next_step_valuation
        return out
    return valuation



class DPN:  #ANN with Pytorch
    def __init__(self, enve):

        self.n_inputs = len(enve.filled.flatten())
        #TODO: Make outputs reflexive
        self.n_outputs = 11
        self.env = enve

        self.rewardsAVG = []
        self.rewardsMAX = []


        # Define network
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("Running on the GPU")
        else:
            self.device = torch.device("cpu")
            print("Running on the CPU")

        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, 128).cuda(),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, self.n_outputs).cuda(),
            nn.Softmax())

        #self.network = torch.load('284_schds.pt')
        self.network.to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=1e-3)

    def predict(self, state):

        state = state.flatten()
        action_probs = self.network(torch.FloatTensor(state).to(self.device))
        return action_probs

        '''
        all caps words are hyperparameters you would set.
        '''

    def train(self, ITERATIONS):
        model = self.network
        opt = self.optimizer
        check = torch.load('114_schds.pt')

        model.load_state_dict(check['model'])
        opt.load_state_dict(check['opt'])

        self.network = model
        self.optimizer = opt

        self.cnt = 0
        for i in range(570,ITERATIONS):
            self.cnt += 1
            self.train_on_jobs()
            print("Iteration " + str(i+1) + " Completed with avg reward: " + str(self.rewardsAVG[-1]), "  Loss:" + "   " +  str(self.loss))

            if self.cnt % 5 == 0 or i==570:
                location = "./"+str(i+1)+"_schds.pt"
                torch.save({'model': self.network.state_dict(), 'opt':self.optimizer.state_dict()}, location)
                #print(self.network[4].weight)



    def trajectory(self, current_state_env):
        '''
        Maybe this implementation doesn't utilize GPUs very well, but I have no clue or not.
        Final output looks like:
        [(s_0, a_0, r_0), ..., (s_L, a_L, r_l)]
        '''

        output_history = []
        cn = 0
        while True:

            current_state = current_state_env.filled
            probs = self.predict(current_state)#could be self.predict()   TODO (by model building, or custom implementation). Basically define model architecture
            pa = Categorical(probs)

            picked_action = pa.sample() #returns index of the action/job selected.
            new_state, t  = current_state_env.updateState( picked_action.item(), current_state) #Get the reward and the new state that the action in the environment resulted in. None if action caused death. TODO build in environment
            if t:
                cn += 1
                reward = current_state_env.reward[-1]
                output_history.append( (current_state, picked_action, reward) )

                if cn > 50:

                    break
            else:
                pass
            if new_state is None: #essentially, you died or finished your trajectory
                break
            else:
                current_state = new_state

        return output_history

    def train_on_jobs(self):
        '''
        Training from a batch. Kinda presume the batch is a set of starting states not sure how you have the implemented states (do they include actions internally?)
        example shape of episode_array
        [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 6, 7],
        [1, 2, 3]
        ]
        '''

        self.optimizer.zero_grad() #This sets the optimizer to update the weights by 0. We'll add to it over time! TODO: Check that it actually works
        for job_start in range(50):
            #episode_array is going to be an array of length N containing trajectories [(s_0, a_0, r_0), ..., (s_L, a_L, r_0)]

            self.envi = CE(70)
            episode_array = [self.trajectory(self.envi) for x in range(EPISODES)]
            # Now we need to make the valuations
            longest_trajectory = 50
            valuation_fun = curried_valuation(longest_trajectory)
            cum_values = np.array([valuation_fun(ep) for ep in episode_array])
            #can compute baselines without a loop?
            baseline_array = np.array([sum(cum_values[:,i])/EPISODES for i in range(longest_trajectory)]) #Probably defeats the purpose of numpy, but we're essentially trying to sum each valuation array together, and then divide by the number of episodes TODO make it work nicely

            #self.rewards.append(baseline_array[-1])
            self.rewardsAVG.append(sum(baseline_array)/len(baseline_array))

            for i in range(EPISODES): #swapped two for loops
                for t in range(50):
                    try:
                        state, action, reward = episode_array[i][t]
                    except IndexError: #this occurs when the trajectory died
                        break
                    #get probabilities from the network. We already did this, but pretty sure we gotta do it again.
                    probs = self.predict(state)
                    DPN_Theta = Categorical(probs) #Pytorch distribution for Categorical classes. SHOULD connect to the network to update weights.
                    if i == 0 and t == 0: #Define the first loss in the sum
                        self.loss = -(cum_values[i][t]-baseline_array[t])*ALPHA*DPN_Theta.log_prob(action)
                    else: #Keep adding to the loss
                        self.loss += -(cum_values[i][t]-baseline_array[t])*ALPHA*DPN_Theta.log_prob(action) #This is what it _should_ look like in pytorch. Added negative (trying to maximize reward, but we're trying to find a minimum) on recommendation of pytorch documentation: https://pytorch.org/docs/stable/distributions.html
        self.loss.backward() #Compute the total cumulated gradient thusfar through our big-ole sum of losses
        self.optimizer.step() #Actually update our network weights. The connection between loss and optimizer is "behind the scenes", but recall that it's dependent

    def point_pred(self, current_state, r):
        actionsss = []
        probs = self.predict(current_state)#could be self.predict()   TODO (by model building, or custom implementation). Basically define model architecture
        if r:
            pa = Categorical(probs)
            picked_action = pa.sample()
        else:
            picked_action = probs.argmax()
        act = picked_action.item()

        return(act)


class DP_CNN:  #CNN with Pytorch
    def __init__(self, enve):
        self.input = enve.filled
        self.dims = self.input.shape
        #TODO: Make outputs reflexive
        self.n_outputs = 11
        self.env = enve
        self.n_flatten = int(8*(int(self.dims[0]/4))*int((self.dims[1]/4)))
        self.rewards = []
        # Define network
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("Running on the GPU")
        else:
            self.device = torch.device("cpu")
            print("Running on the CPU")
        self.network = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Adding the final linear layer
            nn.Flatten(),
            nn.Linear(self.n_flatten, 128),
            nn.Linear(128, self.n_outputs),
            nn.Softmax())
        self.network.to(self.device)


    # def forward(self, x):
    #     x = self.cnn_layers(x)
    #     x = x.view(x.size(0), -1)
    #     x = self.linear_layers(x)
    #     return x

    # Defining the forward pass
    def predict(self, state):
        state = state.reshape(1,1,self.dims[0], self.dims[1])
        action_probs = self.network(torch.tensor(state, dtype=torch.float).to(self.device))
        return action_probs


                               # key looks like (state, action) value is the probability? marked with optional1
    def train(self, ITERATIONS):
        optimizer = optim.Adam(self.network.parameters(), lr=1e-3)
        cnt = 0
        for i in range(ITERATIONS):

            cnt += 1
            self.train_on_jobs(optimizer)

            print("Iteration " + str(i+1) + " Completed with reward: " + str(self.rewards[-1])) #+ " Variance of :" + str(self.variance[-1]))

            if cnt % 100 == 0:
                location = "./"+str(i)+"_schds.pt"
                torch.save(self.network.state_dict(), location)



    def trajectory(self, current_state_env):
        '''
        Maybe this implementation doesn't utilize GPUs very well, but I have no clue or not.
        Final output looks like:
        [(s_0, a_0, r_0), ..., (s_L, a_L, r_l)]
        '''

        output_history = []
        cn = 0
        while True:
            cn += 1
            current_state = current_state_env.filled
            probs = self.predict(current_state)#could be self.predict()   TODO (by model building, or custom implementation). Basically define model architecture
            pa = Categorical(probs)
            picked_action = pa.sample() #returns index of the action/job selected.
            #self.prob_history[(current_state, picked_action)] = choice_prob #optional1
            new_state = current_state_env.updateState( picked_action.item(), current_state) #Get the reward and the new state that the action in the environment resulted in. None if action caused death. TODO build in environment

            reward = current_state_env.reward


            output_history.append( (current_state, picked_action, reward) )

            if cn > 50:
                break
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
        for job_start in range(50):
            #episode_array is going to be an array of length N containing trajectories [(s_0, a_0, r_0), ..., (s_L, a_L, r_0)]

            self.envi = CE(70)
            episode_array = [self.trajectory(self.envi) for x in range(EPISODES)]
            # Now we need to make the valuations
            longest_trajectory = 50
            valuation_fun = curried_valuation(longest_trajectory)
            cum_values = np.array([valuation_fun(ep) for ep in episode_array])
            #can compute baselines without a loop?
            baseline_array = np.array([sum(cum_values[:,i])/EPISODES for i in range(longest_trajectory)]) #Probably defeats the purpose of numpy, but we're essentially trying to sum each valuation array together, and then divide by the number of episodes TODO make it work nicely
            self.rewards.append(baseline_array[0])

            for i in range(EPISODES): #swapped two for loops
                for t in range(50):
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



#cnn training
"""
dpn2 = DP_CNN(CE(70))

dpn2.train(1)
"""
