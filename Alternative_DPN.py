"""
Author: Andrew Reilly, andrew.reilly19@ncf.edu
Date: 11/18/20
Purpose:
    This file is an alternative DPN for me to work from, pulling in information from the original DPN file and outside sources.
"""


import numpy as np
import random
import torch
from torch.distributions import Categorical
from torch import nn, optim
import sys

#print("PyTorch:\t{}".format(torch.__version__))


ITERATIONS = 1000 #kinda like epochs?
BATCH_SIZE = 10   #Might be the exact same thing as episodes, up for interpretation.
EPISODES = 20     #How many trajectories to explore for a given job. Essentually to get a better estimate of the expected reward.
DISCOUNT = 0.99   #how much to discount the reward
ALPHA = 0.001     #learning rate?


class policy_estimator():
    def __init__(self, env):
        self.n_inputs = len(env.flatten())
        #TODO: Make outputs reflexive
        self.n_outputs = 11

        # Define network
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, self.n_outputs))

    def predict(self, state):
        state = state.flatten()
        action_probs = self.network(torch.FloatTensor(state))
        return action_probs

def discount_rewards(rewards, gamma=0.99):
    rwd = np.array([gamma**i * rewards[i]
        for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    rwd = rwd[::-1].cumsum()[::-1]
    return (rwd - rwd.mean())

###update code below here
def reinforce(env, policy_estimator, num_episodes=2000,
              batch_size=10, gamma=0.99):
    # Set up lists to hold results
    total_rewards = []
    batch_rewards = []
    batch_actions = []
    batch_states = []
    batch_counter = 1

    # Define optimizer
    optimizer = optim.Adam(policy_estimator.network.parameters(), lr=0.01)

    action_space = np.arange(env.action_space.n)
    ep = 0
    while ep < num_episodes:
        s_0 = env.reset()
        states = []
        rewards = []
        actions = []
        done = False
        while done == False:
            # Get actions and convert to numpy array
            action_probs = policy_estimator.predict(
                s_0).detach().numpy()
            action = np.random.choice(action_space,
                p=action_probs)
            s_1, r, done, _ = env.step(action)

            states.append(s_0)
            rewards.append(r)
            actions.append(action)
            s_0 = s_1

            # If done, batch data
            if done:
                batch_rewards.extend(discount_rewards(
                    rewards, gamma))
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_counter += 1
                total_rewards.append(sum(rewards))

                # If batch is complete, update network
                if batch_counter == batch_size:
                    optimizer.zero_grad()
                    state_tensor = torch.FloatTensor(batch_states)
                    reward_tensor = torch.FloatTensor(
                        batch_rewards)
                    # Actions are used as indices, must be
                    # LongTensor
                    action_tensor = torch.LongTensor(
                       batch_actions)

                    # Calculate loss
                    logprob = torch.log(
                        policy_estimator.predict(state_tensor))
                    selected_logprobs = reward_tensor * \
                        torch.gather(logprob, 1,
                        action_tensor).squeeze()
                    loss = -selected_logprobs.mean()

                    # Calculate gradients
                    loss.backward()
                    # Apply gradients
                    optimizer.step()

                    batch_rewards = []
                    batch_actions = []
                    batch_states = []
                    batch_counter = 1

                avg_rewards = np.mean(total_rewards[-100:])
                # Print running average
                print("\rEp: {} Average of last 100:" +
                     "{:.2f}".format(
                     ep + 1, avg_rewards), end="")
                ep += 1

    return total_rewards



