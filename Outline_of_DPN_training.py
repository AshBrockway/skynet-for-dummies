"""
WARNING:

Code will not be executable from this alone. Must clean up the TODOs.

Provides a backbone for the training method on a deep policy network.

Presumably, the input to the network will be images as the current state (including current resource usage, and random M jobs to schedule).
The output predictions are the probabilities for choosing to schedule one of the M jobs.

Most important is the train_on_jobs method for updating the weights of the network, in accordance with DPN loss functions.
"""


import numpy as np
from random
from torch.distributions import Categorical
from torch import optim


ITERATIONS = 1000 #kinda like epochs?
BATCH_SIZE = 10   #Might be the exact same thing as episodes, up for interpretation.
EPISODES = 20     #How many trajectories to explore for a given job. Essentually to get a better estimate of the expected reward.
DISCOUNT = 0.99   #how much to discount the reward
ALPHA = 0.001     #learning rate?

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
        assert x.ndim >= 1
        return out
    return valuation


class DPN:#(keras_module or whatever):
    def __init(self)__:
        #Super(self, __init__) #Initialize base methods of keras NN module stuff?
        '''
        define your shit about the NN stuff initial weights, architecture, and so forth
        WE NEED TO FIGURE OUT HOW TO GET GRADIENT of log(policy(state, action))
        Also allow weights to be updated through addition? Something like that.


        INPUT_SIZE = size and shape from the environment's output for a state  TODO
        OUTPUT_SIZE = number of possible actions                               TODO

        Probably include stuff to interact with the environment after inputting a class

        all caps words are hyperparameters you would set.
        '''
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
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3) #This kind of definition might actually have to be defined outside of our model class. TODO 1/2: split model definition from training functions. Half for readability, half to work???
        for i in range(ITERATIONS):
            resource_constraints, time = self.env.generate_random_jobs()
            jobs = self.env.make_starting_states(resource_constraints, time) #this would be a list of starting states
            self.train_on_jobs(jobs, optimizer)

    def predict(self, state):
        '''
        The forward pass of the network on the given state. Returns the output probabilites for taking the OUTPUT_SIZE probabilites

        might already be defined from the initialization after defining your model

        TODO: Rename to forward(self, state), and output appropriate pytorch SHIT
        '''
        pass


    def trajectory(self, current_state, refresh_defaults = True, output_history = []):
        '''
        Maybe this implementation doesn't utilize GPUs very well, but I have no clue or not.

        Final output looks like:
        [(s_0, a_0, r_0), ..., (s_L, a_L, r_l)]
        '''
        if refresh_defaults:
            output_history = []
        probs = self.forward(current_state)#could be self.predict()   TODO (by model building, or custom implementation). Basically define model architecture
        picked_action = Categorical(probs).sample() #returns index of the action/job selected.
        #self.prob_history[(current_state, picked_action)] = choice_prob #optional1
        new_state, reward = self.env.state_and_reward(current_state, picked_action) #Get the reward and the new state that the action in the environment resulted in. None if action caused death. TODO build in environment
        output_history.append( (current_state, picked_action, reward) )
        if new_state is None: #essentially, you died or finished your trajectory
            return output_history
        return  self.trajectory(new_state, False, output_history)

    def train_on_jobs(self,jobset, optimizer):
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
        for job_start in jobset:
            #episode_array is going to be an array of length N containing trajectories [(s_0, a_0, r_0), ..., (s_L, a_L, r_0)]
            episode_array = [self.trajectory(job_start) for x in range(EPISODES)]
            # Now we need to make the valuations
            longest_trajectory = max(len(episode) for episode in episode_array)
            valuation_fun = curried_valuation(longest_trajectory)
            cum_values = np.array(map(episode_array, valuation_fun)) #should be a EPISODESxlength sized
            #can compute baselines without a loop?
            baseline_array = np.array([sum(cum_values[:,i])/EPISODES for i in range(longest_trajectory)]) #Probably defeats the purpose of numpy, but we're essentially trying to sum each valuation array together, and then divide by the number of episodes TODO make it work nicely
            for i in range(EPISODES): #swapped two for loops
                for t in range(longest_trajectory):
                    try:
                        state, action, reward = episodes_array[i][t]
                    except IndexError: #this occurs when the trajectory died
                        break
                    #get probabilities from the network. We already did this, but pretty sure we gotta do it again.
                    probs = self.forward(state)
                    DPN_Theta = Categorical(probs) #Pytorch distribution for Categorical classes. SHOULD connect to the network to update weights.
                    if i == 0 and t == 0: #Define the first loss in the sum
                        loss = -(cum_values[i][t]-baseline_array[t])*ALPHA*DPN_Theta.log_prob(action)
                    else: #Keep adding to the loss
                        loss += -(cum_values[i][t]-baseline_array[t])*ALPHA*DPN_Theta.log_prob(action) #This is what it _should_ look like in pytorch. Added negative (trying to maximize reward, but we're trying to find a minimum) on recommendation of pytorch documentation: https://pytorch.org/docs/stable/distributions.html
        loss.backward() #Compute the total cumulated gradient thusfar through our big-ole sum of losses
        optimizer.step() #Actually update our network weights. The connection between loss and optimizer is "behind the scenes", but recall that it's dependent