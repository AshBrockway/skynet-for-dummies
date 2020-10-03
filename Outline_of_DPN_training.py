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
        x = len(episode)
        if x != length_of_longest_trajectory:
            #If the episode isn't as long as the longest trajectory, pad it
            episode.extend([(0,0,0) for x in range(length_of_longest_trajectory-x)]) #have to make sure the numbers line up correctly
        # TODO
        #compute valuation with the episode/trajectory after it's been padded. There could be something clever here.
        out = np.array([valuation for valuation in range(length_of_longest_trajectory)])
        #out = do_the_thing(episode)
        return out
    return valuation


def randomly_selected_action(probs):
    '''
    Given a set of output probabilities corresponding to actions (or jobs to schedule):

    Randomly select one action with the described probabilities.

    Output the index of the job to schedule and the probability that we chose that action.

    For large probability lists, we might choose the final probability less than we expect due to floating point arithmetic problems.
    '''
    action_number = random.uniform(0,1)
    index = 0
    lower = 0
    upper = probs[index]
    while not (lower <= action_number and action_number <= upper):
        index += 1
        lower += upper
        upper += probs[index]
    return index, probs[index]


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
        for i in range(ITERATIONS):
            resource_constraints, time = self.env.generate_random_jobs()
            jobs = self.env.make_starting_states(resource_constraints, time) #this would be a list of starting states
            self.train_on_jobs(jobs)

    def predict(self, state):
        '''
        The forward pass of the network on the given state. Returns the output probabilites for taking the OUTPUT_SIZE probabilites

        might already be defined from the initialization after defining your model
        '''
        pass


    def trajectory(self, current_state, output_history = []):
        '''
        Maybe this implementation doesn't utilize GPUs very well, but I have no clue or not.

        Final output looks like:
        [(s_0, a_0, r_0), ..., (s_L, a_L, r_l)]
        '''
        probs = self.model.predict(current_state)#could be self.predict()   TODO (by model building, or custom implementation). Basically define model architecture
        picked_action, choice_prob = randomly_selected_action(probs) #returns index of the action/job selected.
        self.prob_history[(current_state, picked_action)] = choice_prob #optional1
        new_state, reward = self.env.state_and_reward(current_state, picked_action) #Get the reward and the new state that the action in the environment resulted in. None if action caused death. TODO build in environment
        output_history.append( (current_state, picked_action, reward) )
        if new_state is None: #essentially, you died or finished your trajectory
            return output_history
        return  self.trajectory(new_state, output_history)

    def train_on_jobs(self,jobset):
        '''
        Training from a batch. Kinda presume the batch is a set of starting states not sure how you have the implemented states (do they include actions internally?)

        example shape of episode_array
        [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 6, 7],
        [1, 2, 3]
        ]
        '''
        delta = np.zeros(len(self.weights), shape = self.weights.shape) #Basically start gradient or how you'll change weights out at 0 but with the shape or whatever you need to update the weights through addition. TODO figure out how this thing should look
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
                    #first two products are scalars, final is scalar multiplication of computed gradients on the NN
                    #TODO figure out the shape or how to access the DPN's gradient thingy?
                    #TODO GRADIENT HOW DO???
                    delta += (cum_values[i][t]-baseline_array[t])*ALPHA*
                                self.gradient( Math.log( self.prob_history[(state, action)] ) ) #this might not even make sense in implementation
        self.weights = self.weights + delta
