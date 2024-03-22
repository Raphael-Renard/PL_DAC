
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from Game_representation import Morpion



class DQN:
    def __init__(self, state_channels, action_size):
        self.state_channels = state_channels
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model() 
        self.target_model = self._build_model() 
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()


    def _build_model(self):
        model = nn.Sequential(
        nn.Conv2d(in_channels=self.state_channels, out_channels=32, kernel_size=3, stride=1, padding=0),
        nn.Flatten(),
        nn.ReLU(),
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, self.action_size))

        return model
    
    def remember(self, state, action, reward, next_state, done, valid_action):
        self.memory.append((state, action, reward, next_state, done, valid_action))

    def act(self, env, state, board_x, board_y):
        if np.random.rand() <= self.epsilon:
            return random.choice(env.get_possible_moves())
        
        state = torch.FloatTensor(np.array(state)).unsqueeze(0)
        q_values = self.model(state)[0]

        for move in env.get_possible_moves(): # prend un coup valide
            if move[0]//3 == board_x and move[1]//3 == board_y: # coups légaux dans notre petite grille
                q_values[coordinates_to_index(move[0],move[1])]+=2000
        action = torch.argmax(q_values).item()
        action = index_to_coordinates(action)
        return action[0]+3*board_x,action[1]+3*board_y

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done, valid_action in minibatch:
            state_tensor = torch.FloatTensor(np.array(state)).unsqueeze(0)
            target = reward
            if not done:
                if valid_action:
                    next_state_tensor = torch.FloatTensor(np.array(next_state)).unsqueeze(0)
                    target = (reward + self.gamma * torch.max(self.target_model(next_state_tensor)).item())
                else:
                    # If the action was not valid, there's no next state
                    target = reward
            target_f = self.model(state_tensor).squeeze(0)
            action = coordinates_to_index(action[0], action[1])
            target_f[action] = target
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.model(state_tensor), target_f.unsqueeze(0))
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())


def index_to_coordinates(i):
    x = i // 3
    y = i % 3
    return x, y

def coordinates_to_index(x,y):
    x = x%3
    y = y%3
    return x * 3 + y

"""
def plot_res(values):   
    plt.plot(values, label='score per run')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    x = range(len(values))
    plt.legend()
   
    try:
        z = np.polyfit(x, values, 1)
        p = np.poly1d(z)
        plt.plot(x,p(x),"--", label='trend')
    except:
        print('')

    plt.show()
"""


state_channels = 3  # represent each small grid with 3 channels (one for player 1, one for player 2, one for empty)
action_size = 9  # 9 possible actions (one for each cell in the grid)
agent = DQN(state_channels, action_size)
C =10
final = []

# Training loop
batch_size = 32
num_episodes = 500
for e in range(num_episodes):
    total_reward = 0
    done = False
    env = Morpion(boards=np.zeros((3, 3, 3, 3), dtype=int),
                  big_boards=np.zeros((3, 3), dtype=int),
                  empty_boards=[[[(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)],
                    [(0,3),(0,4),(0,5),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5)],
                    [(0,6),(0,7),(0,8),(1,6),(1,7),(1,8),(2,6),(2,7),(2,8)]],
                    [[(3,0),(3,1),(3,2),(4,0),(4,1),(4,2),(5,0),(5,1),(5,2)],
                    [(3,3),(3,4),(3,5),(4,3),(4,4),(4,5),(5,3),(5,4),(5,5)],
                    [(3,6),(3,7),(3,8),(4,6),(4,7),(4,8),(5,6),(5,7),(5,8)]],
                    [[(6,0),(6,1),(6,2),(7,0),(7,1),(7,2),(8,0),(8,1),(8,2)],
                    [(6,3),(6,4),(6,5),(7,3),(7,4),(7,5),(8,3),(8,4),(8,5)],
                    [(6,6),(6,7),(6,8),(7,6),(7,7),(7,8),(8,6),(8,7),(8,8)]]],
                  empty_all={(i, j) for i in range(9) for j in range(9)})
    while not done:
        (i,j) = env.get_possible_moves()[0] #
        state,board_x,board_y = env.get_grid(i,j)
        action = agent.act(env, state, board_x, board_y)
        if action not in env.get_possible_moves():
            print("Action non valide :", action)
            reward = -100
            total_reward += reward
            done = True
            valid_action = False
            next_state = None
        else:
            next_state, reward, done = env.step(action)
            total_reward += reward
            if done:
                agent.update_target_model()
                print("episode: {}/{}, score:{}, e: {:.2}".format(e+1, num_episodes, total_reward, agent.epsilon))
                break
            board_x,board_y = next_state[1],next_state[2]
            next_state = next_state[0]
            valid_action = True
        agent.remember(state, action, reward, next_state, done, valid_action)
        state = next_state
        if done:
            agent.update_target_model()
            print("episode: {}/{}, reward:{}, e: {:.2}".format(e+1, num_episodes, reward, agent.epsilon))
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        if e % C == 0:
                agent.update_target_model()

    final.append(total_reward)
#plot_res(final)
