import numpy as np
import random
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )
        return model

    def forward(self, x):
        return self.model(x)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(state)
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.forward(state)
        return np.argmax(act_values.detach().numpy())

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                target = (reward + self.gamma * torch.max(self.forward(next_state_tensor)).item())
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            target_f = self.forward(state_tensor).squeeze(0).detach().numpy()
            target_f[action] = target
            target_f = torch.FloatTensor(target_f)
            self.model.train()
            self.model.zero_grad()
            loss = nn.MSELoss()(self.forward(state_tensor), target_f.unsqueeze(0))
            loss.backward()
            optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class GameState():
    def __init__(self):
        self.player = 1
        self.last_move = None

    def get_possible_moves(self):
        pass

    def make_move(self, move):
        pass

    def is_terminal(self):
        pass

    def get_result(self):
        pass


class MorpionDQN(GameState):
    def __init__(self, boards, big_boards, empty_boards, empty_all):
        super().__init__()
        self.boards = boards
        self.big_boards = big_boards
        self.empty_boards = empty_boards
        self.empty_all = empty_all
        self.state_size = len(empty_all) * 2
        self.action_size = len(empty_all)

    def get_possible_moves(self):
        if self.last_move is not None:
            board_x = self.last_move[0] % 3
            board_y = self.last_move[1] % 3
            if self.empty_boards[board_x][board_y] != []:
                return self.empty_boards[board_x][board_y]
        return list(self.empty_all)

    def make_move(self, move):
        i, j = move
        big_board_x = i // 3
        big_board_y = j // 3
        x = i % 3
        y = j % 3
        new_state = MorpionDQN(
            boards=np.copy(self.boards),
            empty_all=self.empty_all.copy(),
            empty_boards=copy.deepcopy(self.empty_boards),
            big_boards=np.copy(self.big_boards)
        )
        new_state.boards[big_board_x, big_board_y, x, y] = self.player
        new_state.empty_all.remove((i, j))
        new_state.empty_boards[big_board_x][big_board_y].remove((i, j))

        if new_state.is_a_board_completed(big_board_x, big_board_y, x, y):
            new_state.big_boards[big_board_x, big_board_y] = self.player
            new_state.empty_all -= {(x + 3 * big_board_x, y + 3 * big_board_y) for x in range(3) for y in range(3)}
            new_state.empty_boards[big_board_x][big_board_y] = []

        new_state.player = -self.player
        new_state.last_move = move
        return new_state

    def is_terminal(self, move):
        if abs(self.big_boards[move[0]].sum()) == 3 or abs(self.big_boards[:, move[1]].sum()) == 3:
            return True

        if move[0] + move[1] % 2 == 0:
            if abs(self.big_boards[0, 0] + self.big_boards[1, 1] + self.big_boards[2, 2]) == 3 \
                    or abs(self.big_boards[2, 0] + self.big_boards[1, 1] + self.big_boards[0, 2]) == 3:
                return True

        if not self.empty_all:
            return True
        return False

    def is_a_board_completed(self, board_x, board_y, x, y):
        if abs(self.boards[board_x][board_y][x].sum()) == 3 or abs(self.boards[board_x][board_y][:, y].sum()) == 3:
            return True

        if (x + y) % 2 == 0:
            if abs(self.boards[board_x][board_y][0, 0] + self.boards[board_x][board_y][1, 1] +
                   self.boards[board_x][board_y][2, 2]) == 3 \
                    or abs(self.boards[board_x][board_y][2, 0] + self.boards[board_x][board_y][1, 1] +
                            self.boards[board_x][board_y][0, 2]) == 3:
                return True
        if self.empty_boards[board_x][board_y] == []:
            return True
        return False

    def make_move_self(self, move):
        i, j = move
        big_board_x = i // 3
        big_board_y = j // 3
        x = i % 3
        y = j % 3
        self.boards[big_board_x, big_board_y, x, y] = self.player
        self.empty_all.remove((i, j))
        self.empty_boards[big_board_x][big_board_y].remove((i, j))

        if self.is_a_board_completed(big_board_x, big_board_y, x, y):
            self.big_boards[big_board_x, big_board_y] = self.player
            self.empty_all -= {(x + 3 * big_board_x, y + 3 * big_board_y) for x in range(3) for y in range(3)}
            self.empty_boards[big_board_x][big_board_y] = []
            self.player = -self.player
            self.last_move = move
            return self.is_terminal((big_board_x, big_board_y))

        self.player = -self.player
        self.last_move = move
        return False

    def get_result(self):
        for i in range(3):
            if self.big_boards[i].sum() == 3 * self.player or self.big_boards[:, i].sum() == 3 * self.player:
                return self.player
        if (self.big_boards[0, 0] == self.player and self.big_boards[1, 1] == self.player and self.big_boards[
            2, 2] == self.player) or \
                (self.big_boards[2, 0] == self.player and self.big_boards[1, 1] == self.player and self.big_boards[
                    0][2] == self.player):
            return self.player
        else:
            return 0
    
    def calculate_reward(self,action):
        if self.is_terminal((action[0]//3,action[1]//3)):
            return self.get_result()
        else:
            return 0

    def step(self, action):
        # Apply the action to the environment and obtain the next state, reward, and whether the episode is done
        self.make_move_self(action)
        reward = self.calculate_reward(action)  #
        done = self.is_terminal((action[0]//3,action[1]//3))
        return reward, done


# Exemple d'utilisation
env = MorpionDQN(boards=np.zeros((3, 3, 3, 3), dtype=int),
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

state_size = env.state_size
action_size = env.action_size
agent = DQN(state_size, action_size)

optimizer = optim.Adam(agent.parameters(), lr=0.001)
batch_size = 32
num_episodes = 1000
for e in range(num_episodes):
    state = env.get_possible_moves()  #
    for time in range(500):
        action = agent.act(state)
        reward, done = env.step(action)
        next_state = env.get_possible_moves()
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print("episode: {}/{}, score: {}, e: {:.2}".format(e, num_episodes, time, agent.epsilon))
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
