import math
import random
import numpy as np
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


#AGENT DQN 


def build_model(input_shape, action_space):
        model = nn.Sequential(
        nn.Linear(input_shape, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, action_space)
        )
        return model
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95   
        self.epsilon = 1.0  
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.955
        self.model = build_model(state_size, action_size)
        self.target_model = build_model(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if torch.rand(1) <= self.epsilon:
            return torch.randint(0, self.action_size, (1,)).item()
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()

    def replay(self, batch_size,verbose = True):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states, targets = [], []
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)

            target = self.model(state_tensor)
            if done:
                target[action] = reward
            else:
                target[action] = reward + self.gamma * torch.max(self.target_model(next_state_tensor))

            states.append(state)
            targets.append(target)

        states_tensor = torch.tensor(states, dtype=torch.float32)
        targets_tensor = torch.stack(targets)
        
        self.optimizer.zero_grad()
        outputs = self.model(states_tensor)
        loss = self.loss_fn(outputs, targets_tensor)
        loss.backward()
        self.optimizer.step()


        if(verbose) : 
            print(f"Loss: {loss.item()}")
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())



#ENVIRONNEMENT         


def distance_to(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def angle_to(p1, p2):
    d = distance_to(p1, p2)
    if d == 0:
        d = .1
    dx = (p2[0] - p1[0]) / d
    dy = (p2[1] - p1[1]) / d

    a = int(math.acos(dx) * 180 / math.pi)

    if dy < 0:
        a = 360 - a

    return a


def diff_angle(a1, a2):
    right = a2 - a1 if a1 <= a1 else 360 - a1 + a2
    left = a1 - a2 if a1 >= a2 else a1 + 360 - a2

    return right if right < left else -left


class MadPodRacing:
    def __init__(self):
        self.nb_actions = 56
        self.nb_states = 4
        self.player_pos = pygame.Vector2(random.uniform(1500, 14500), random.uniform(1500, 7500))
        self.checkpoint_pos = pygame.Vector2(random.uniform(1500, 14500), random.uniform(1500, 7500))
        self.angle = random.randint(0, 359)
        self.velocity = pygame.Vector2(random.uniform(0, 400), random.uniform(0, 400))
        self.prev_thrust = 0
        self.player_state = self.get_state()

    def reset(self):
        self.checkpoint_pos = pygame.Vector2(random.randint(1500, 14500), random.randint(1500, 7500))
        self.player_pos = pygame.Vector2(random.randint(1500, 14500), random.randint(1500, 7500))
        
        while self.player_pos.distance_to(self.checkpoint_pos) < 800:
            self.player_pos = pygame.Vector2(random.randint(1500, 14500), random.randint(1500, 7500))
        self.angle = random.randint(0, 359)
        self.velocity = pygame.Vector2(random.randint(0, 400), random.randint(0, 400))
        self.player_state = self.get_state()
        return self.player_state

    def get_state(self):
        distance_to_checkpoint = self.checkpoint_pos.distance_to(self.player_pos)

        checkpoint_angle = angle_to(self.player_pos, self.checkpoint_pos)
        angle_to_checkpoint = diff_angle(self.angle, checkpoint_angle)

        velocity_length = int(self.velocity.length())

        velocity_angle = angle_to(self.player_pos, self.player_pos + self.velocity)
        angle_to_velocity = diff_angle(self.angle, velocity_angle)

        return distance_to_checkpoint, angle_to_checkpoint, velocity_length, angle_to_velocity


    """def step(self, action, verbose=False):
        angles = [-18, 0, 18]
        dthrust = [-15, 0, 15]

        thrust = self.prev_thrust + dthrust[action // 3]
        thrust = max(0, min(100, thrust))
        self.prev_thrust = thrust

        angle = (self.angle + angles[action % 3]) % 360

        target_x = self.player_pos.x + 10000 * math.cos(math.radians(angle))
        target_y = self.player_pos.y + 10000 * math.sin(math.radians(angle))

        dt = 1

        a = angle_to(self.player_pos, (target_x, target_y))

        right = a - self.angle if self.angle <= a else 360 - self.angle + a
        left = self.angle - a if self.angle >= a else self.angle + 360 - a

        da = right if right < left else -left

        if da > 18:
            da = 18
        elif da < -18:
            da = -18

        self.angle += da

        if self.angle >= 360:
            self.angle -= 360
        elif self.angle < 0:
            self.angle += 360

        ra = self.angle * math.pi / 180

        self.velocity.x += math.cos(ra) * thrust
        self.velocity.y += math.sin(ra) * thrust

        if verbose:
            print("player_pos before step: ", self.player_pos)

        self.player_pos += self.velocity * dt

        self.player_pos.x = round(self.player_pos.x)
        self.player_pos.y = round(self.player_pos.y)

        if verbose:
            print("player_pos after step: ", self.player_pos)

        self.velocity.x = int(self.velocity.x * .85)
        self.velocity.y = int(self.velocity.y * .85)

        distance_to_checkpoint = self.player_pos.distance_to(self.checkpoint_pos)

        if verbose:
            print("distance_to_checkpoint :", distance_to_checkpoint)

        if distance_to_checkpoint < 800:
            reward = 100
        else:
            reward = -distance_to_checkpoint * 23

        if distance_to_checkpoint < 800:
            done = True
        else:
            done = False

        next_state = self.get_state()
        return next_state, reward, done"""
        
    def step(self, action, verbose=False):
        
        angle = [0, 45, 90, 135, 180, 225, 270, 315]
        dthrust = [0, 15, 30, 45, 60, 75, 90]

        angle_index = action // len(dthrust)
        thrust_index = action % len(dthrust)

        angle = angle[angle_index]
        thrust = dthrust[thrust_index]

        target_x = self.player_pos.x + 10000 * math.cos(math.radians(angle))
        target_y = self.player_pos.y + 10000 * math.sin(math.radians(angle))

        dt = 1

        a = angle_to(self.player_pos, (target_x, target_y))

        right = a - self.angle if self.angle <= a else 360 - self.angle + a
        left = self.angle - a if self.angle >= a else self.angle + 360 - a

        da = right if right < left else -left

        if da > 18:
            da = 18
        elif da < -18:
            da = -18

        self.angle += da

        if self.angle >= 360:
            self.angle -= 360
        elif self.angle < 0:
            self.angle += 360

        ra = self.angle * math.pi / 180

        self.velocity.x += math.cos(ra) * thrust
        self.velocity.y += math.sin(ra) * thrust

        if verbose:
            print("player_pos before step: ", self.player_pos)

        self.player_pos += self.velocity * dt

        self.player_pos.x = round(self.player_pos.x)
        self.player_pos.y = round(self.player_pos.y)

        if verbose:
            print("player_pos after step: ", self.player_pos)

        self.velocity.x = int(self.velocity.x * .85)
        self.velocity.y = int(self.velocity.y * .85)

        distance_to_checkpoint = self.player_pos.distance_to(self.checkpoint_pos)

        if verbose:
            print("distance_to_checkpoint :", distance_to_checkpoint)

        if distance_to_checkpoint < 800:
            reward = 100
        else:
            reward = -distance_to_checkpoint * 23

        if distance_to_checkpoint < 800:
            done = True
        else:
            done = False

        next_state = self.get_state()
        return next_state, reward, done

    


#ENTRAINEMENT
    
env = MadPodRacing()

state_size = env.nb_states
action_size = env.nb_actions
batch_size = 64
n_episodes = 1000
C = 30  # Fréquence de mise à jour du réseau cible


agent = DQNAgent(state_size, action_size)

for episode in range(n_episodes):
    state = env.reset()
    #print("state",state)
    print("Position du player :",env.player_pos)
    print("Position du checkpoint :",env.checkpoint_pos)

    total_reward = 0
    done = False

    while not done:
        action = agent.act(state)
        print("action",action)
        next_state, reward, done = env.step(action)
        print(f"next state : {next_state}, reward : {reward}, done : {done}")
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            agent.update_target_model()
            print(f"Episode: {episode+1}/{n_episodes}, score: {total_reward}, e: {agent.epsilon:.2}")
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

        if episode % C == 0:
            agent.update_target_model()