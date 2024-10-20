import pygame
import sys
import os
import random
import gym
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import mpr_training_env
from mpr_training_env import Env
import matplotlib.pyplot as plt
import pickle

#AGENT
class DQNAgent:
    def __init__(self, state_size, action_size, memory_size=1000000, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.model = self._build_model(state_size, action_size)
        self.target_model = self._build_model(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def _build_model(self, input_shape, action_space):
        model = nn.Sequential(
            nn.Linear(input_shape, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, action_space)
        )
        return model
    

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if torch.rand(1) <= self.epsilon:
            return torch.randint(0, self.action_size, (1,)).item()
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()
          

    """def replay(self, batch_size):

        if len(self.memory) < batch_size:
            return 0
            
        minibatch = random.sample(self.memory, batch_size)
        states = torch.zeros(batch_size, self.state_size)
        targets = torch.zeros(batch_size, self.action_size)

        total_loss = 0
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            state_tensor = torch.FloatTensor(state)
            next_state_tensor = torch.FloatTensor(next_state)

            target = reward

            if not done:
                target = reward + self.gamma * torch.max(self.target_model(next_state_tensor)).item()

            states[i] = state_tensor
            targets[i][action] = target

        self.optimizer.zero_grad()

        outputs = self.model(states)
        loss = self.loss_fn(outputs, targets)
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()"""
    
    def replay(self, batch_size):

        if len(self.memory) < batch_size:
            return 0
            
        minibatch = random.sample(self.memory, batch_size)
        states = torch.zeros(batch_size, self.state_size)
        targets = torch.zeros(batch_size, self.action_size)

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            state_tensor = torch.FloatTensor(state)
            next_state_tensor = torch.FloatTensor(next_state)

            target = reward

            if not done:
                target = reward + self.gamma * torch.max(self.target_model(next_state_tensor)).item()

            states[i] = state_tensor
            targets[i][action] = target

        self.optimizer.zero_grad()

        outputs = self.model(states)
        loss = self.loss_fn(outputs, targets)
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())


    def pickle(self):
        weights = []
        biases = []
        # Parcourir les paramètres du modèle principal
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                weights.append(param.cpu().detach().numpy())  # Convertir les tensors PyTorch en arrays NumPy
            elif 'bias' in name:
                biases.append(param.cpu().detach().numpy())
        # Sérialiser les poids et les biais à l'aide de pickle
        serialized_data = pickle.dumps((weights, biases))
        return serialized_data





def create_agent(state_size, action_size, memory_size=1000000, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.001):
    return DQNAgent(state_size, action_size, memory_size, gamma, epsilon, epsilon_min, epsilon_decay, learning_rate)



#ENTRAINEMENT

def train_agent(env, agent, n_episodes=100, batch_size=64, C=150, verbose=False):
    
    train_total_loss = []
    train_total_reward = []

    for episode in range(n_episodes):
        state = env.reset()
        if verbose:
            print("Episode : ", episode + 1)
            print("Position du player :", env.player_pos)
            print("Position du checkpoint :", env.checkpoint_pos)

        total_reward = 0
        done = False
        total_loss = 0
        nb_loss = 0
        iteration = 0
        while not done:

            if verbose:
                print("itération: ", iteration)
            iteration += 1

            action = agent.act(state)
            if verbose:
                print("action", action)

            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            if verbose:
                print("Score: {:.15f}".format(reward))
            state = next_state
            total_reward += reward

            if done:
                agent.update_target_model()
                print(f"Episode: {episode+1}/{n_episodes}, total_reward: {total_reward:.6f}, e: {agent.epsilon:.2}")
                break

            if len(agent.memory) > batch_size:
                loss = agent.replay(batch_size)
                if verbose:
                    print("loss: ", loss)
                    print("\n")
                total_loss += loss
                nb_loss += 1 

            if episode % C == 0:
                agent.update_target_model()

        if nb_loss != 0:
            total_loss = total_loss / nb_loss
        train_total_loss .append(total_loss)
        train_total_reward.append(total_reward)

    return train_total_loss, train_total_reward

#GRAPHES

def plot_training_results(train_loss, train_total_reward):
    plt.figure(figsize=(12, 6))

    # Plotting training loss
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Training Loss', color='blue')
    plt.title('Training Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Plotting total rewards
    plt.subplot(1, 2, 2)
    plt.plot(train_total_reward, label='Total Reward', color='green')
    plt.title('Total Reward')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    
