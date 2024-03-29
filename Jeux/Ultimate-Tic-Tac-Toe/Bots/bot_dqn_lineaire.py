import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from Game_representation import Morpion



class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000000) #2000
        self.gamma = 0.99  # discount rate
        self.epsilon = 0.9  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99
        self.model = self._build_model() 
        self.target_model = self._build_model() 
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00005)
        # lr = 0.00005
        # lr = 0.001
        self.loss_fn = nn.MSELoss()


    def _build_model(self):
        model = nn.Sequential(
        nn.Linear(self.state_size, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, self.action_size))

        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    
    def act(self, env, state):
        if torch.rand(1) <= self.epsilon:
            return env.get_possible_moves()[torch.randint(high=len(env.get_possible_moves()),size=(1,))]
        
        state = torch.FloatTensor(np.array(state))
        q_values = self.model(state)[0]

        # Gestion des coups illégaux
        for move in env.get_possible_moves(): # prend un coup valide
            q_values[coordinates_to_index(move)]+=2000
        
        action = torch.argmax(q_values).item()
        action = index_to_coordinates(action)
        return action

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return 0
        
        indices = torch.randint(0, len(self.memory), (batch_size,))
        minibatch = [self.memory[idx] for idx in indices]
        states = torch.zeros(batch_size, self.state_size)
        target_f = torch.zeros((batch_size, self.action_size))

        total_loss = 0
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = torch.FloatTensor(state)
            target = reward

            if not done:
                next_state_tensor = torch.FloatTensor(np.array(next_state))
                target = (reward + self.gamma * torch.max(self.target_model(next_state_tensor)).item())

            action = coordinates_to_index(action)
            target_f[i][action] = target

        self.optimizer.zero_grad()

        loss = self.loss_fn(self.model(states), target_f)
        loss.backward()
        self.optimizer.step()
        total_loss += loss.item()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return total_loss / batch_size

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())



def coordinates_to_index(coordinates):

    x, y = coordinates
    return x * 9 + y

def index_to_coordinates(index):

    x = index // 9
    y = index % 9
    return x, y



state_size = 81  # represent each small grid with 3 channels (one for player 1, one for player 2, one for empty)
action_size = 81  # 9 possible actions (one for each cell in the grid)
agent = DQN(state_size, action_size)
C = 50
train_loss = []



# Training loop
batch_size = 64
num_episodes = 1500

for e in range(num_episodes):
    replay_loss = 0.0
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

        state = np.reshape(env.boards, (1,81))
        action = agent.act(env, state)

        next_state, reward, done = env.step2(action)
        total_reward += reward

        if done:
            agent.update_target_model()
            print("episode: {}/{}, score: {}, eps: {:.2}, loss: {:.6}".format(e+1, num_episodes, total_reward, agent.epsilon,replay_loss))
            break

        agent.remember(state, action, reward, next_state, done)
        state = next_state

        if len(agent.memory) > batch_size:
            replay_loss = agent.replay(batch_size)
        if e % C == 0:
                agent.update_target_model()

    train_loss.append(replay_loss)


plt.plot(train_loss[1:])
plt.title("Loss pendant l'entraînement")
plt.xlabel('Episodes')
plt.ylim(1e-19, 1e-4)
plt.ylabel('Loss')
plt.yscale('log')
plt.show()


plt.plot(train_loss[1:])
plt.title("Loss pendant l'entraînement")
plt.xlabel('Episodes')
plt.ylabel('Loss')
plt.show()

###### Test contre bot aleatoire

from bot_aleatoire import Aleatoire
gagne_dqn = 0
perdu_dqn = 0
neutre_dqn = 0


for partie in range(100):
    env = Morpion(boards=np.array([[[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]]],
                                        [[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]]],
                                        [[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]]]]),
                empty_all={(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2),
                            (0,3),(0,4),(0,5),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5),
                            (0,6),(0,7),(0,8),(1,6),(1,7),(1,8),(2,6),(2,7),(2,8),
                            (3,0),(3,1),(3,2),(4,0),(4,1),(4,2),(5,0),(5,1),(5,2),
                            (3,3),(3,4),(3,5),(4,3),(4,4),(4,5),(5,3),(5,4),(5,5),
                            (3,6),(3,7),(3,8),(4,6),(4,7),(4,8),(5,6),(5,7),(5,8),
                            (6,0),(6,1),(6,2),(7,0),(7,1),(7,2),(8,0),(8,1),(8,2),
                            (6,3),(6,4),(6,5),(7,3),(7,4),(7,5),(8,3),(8,4),(8,5),
                            (6,6),(6,7),(6,8),(7,6),(7,7),(7,8),(8,6),(8,7),(8,8)},
                empty_boards=[[[(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)],
                        [(0,3),(0,4),(0,5),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5)],
                        [(0,6),(0,7),(0,8),(1,6),(1,7),(1,8),(2,6),(2,7),(2,8)]],
                        [[(3,0),(3,1),(3,2),(4,0),(4,1),(4,2),(5,0),(5,1),(5,2)],
                        [(3,3),(3,4),(3,5),(4,3),(4,4),(4,5),(5,3),(5,4),(5,5)],
                        [(3,6),(3,7),(3,8),(4,6),(4,7),(4,8),(5,6),(5,7),(5,8)]],
                        [[(6,0),(6,1),(6,2),(7,0),(7,1),(7,2),(8,0),(8,1),(8,2)],
                        [(6,3),(6,4),(6,5),(7,3),(7,4),(7,5),(8,3),(8,4),(8,5)],
                        [(6,6),(6,7),(6,8),(7,6),(7,7),(7,8),(8,6),(8,7),(8,8)]]],
                big_boards=np.array([[0,0,0],[0,0,0],[0,0,0]]))
    alea = Aleatoire(env)
    T = False

        
    while not T:
        state = np.reshape(env.boards, (1,81))
        action = agent.act(env, state)
        
        env.make_move_self(action)
        T = env.is_terminal((action[0]//3,action[1]//3))
        if T:
            break
        opponent_move = alea.give_move()
        env.make_move_self(opponent_move)
        T = env.is_terminal((opponent_move[0]//3,opponent_move[1]//3))

    resultat = env.get_result()
    if resultat == 1:
        gagne_dqn +=1
    elif resultat == -1:
        perdu_dqn +=1
    else:
        neutre_dqn +=1
    

plt.bar(["gagné","nul","perdu"],[gagne_dqn,neutre_dqn,perdu_dqn],color = ['tab:green', 'tab:blue', 'tab:red'])
plt.ylabel('Nombre de parties')
plt.title('Parties jouées par un agent DQN contre un agent aléatoire')
plt.show()