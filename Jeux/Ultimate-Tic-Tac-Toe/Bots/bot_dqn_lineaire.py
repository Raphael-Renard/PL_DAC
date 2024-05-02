import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from Game_representation import Morpion, MorpionSimple




class DQN:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000000) #2000
        self.gamma = 0.99  # discount rate
        self.epsilon = 0.9  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.5 #0.8
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
            return coordinates_to_index(env.get_possible_moves()[torch.randint(high=len(env.get_possible_moves()),size=(1,))])
        
        state = torch.FloatTensor(np.array(state))
        q_values = self.model(state)[0]

        # Gestion des coups illégaux
        for move in env.get_possible_moves(): # prend un coup valide
            q_values[coordinates_to_index(move)]+=2000
        
        action = torch.argmax(q_values).item()
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


    def train(self, env, nb_episodes, C= 100, batch_size=64, show_plots=False):
        
        train_loss = []
        entropy_values = []


        # Training loop

        for e in range(nb_episodes):
            replay_loss = 0.0
            total_reward = 0
            done = False
            env.reset()

            while not done:

                state = np.reshape(env.boards, (1,81))
                action = agent.act(env, state)

                next_state, reward, done = env.step2(action)
                total_reward += reward

                if done:
                    agent.update_target_model()
                    print("episode: {}/{}, score: {}, eps: {:.2}, loss: {:.6}".format(e+1, nb_episodes, total_reward, agent.epsilon,replay_loss))
                    break

                agent.remember(state, action, reward, next_state, done)
                state = next_state

                if len(agent.memory) > batch_size:
                    replay_loss = agent.replay(batch_size)
                if e % C == 0:
                        agent.update_target_model()

            train_loss.append(replay_loss)


            if show_plots:
                # Calculate entropy
                state_tensor = torch.FloatTensor(state)
                q_values = agent.model(state_tensor)
                action_probs = nn.functional.softmax(q_values, dim=-1)
                    
                entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-8), dim=-1).mean()
                entropy_values.append(entropy.item())

        if show_plots:

            plt.plot(train_loss[3:])
            plt.title("Loss pendant l'entraînement")
            plt.xlabel('Episodes')
            plt.ylabel('Loss')
            plt.show()

            plt.plot(entropy_values[3:])
            plt.title("Entropie pendant l'entraînement")
            plt.xlabel('Episodes')
            plt.ylabel('Entropie')
            plt.show()


def coordinates_to_index(coordinates):
    x, y = coordinates
    return x * 9 + y

def index_to_coordinates(index):
    x = index // 9
    y = index % 9
    return x, y









###### Test contre bot aleatoire

from bot_aleatoire import Aleatoire


liste_gagne = []
liste_perdu = []

#NB_EPISODES = [100,200,300,400,500,600,700,800]
NB_EPISODES_TRAIN = range(50,500,50)


for nb_episodes in NB_EPISODES_TRAIN:
        
    gagne_dqn = 0
    perdu_dqn = 0
    neutre_dqn = 0

    state_size = 81  # the whole grid
    action_size = 81  # the whole grid
    env = MorpionSimple()
    agent = DQN(state_size, action_size)

    agent.train(env, nb_episodes, state_size, action_size)

    for partie in range(100):
        env.reset()
        alea = Aleatoire(env)
        T = False

            
        while not T:
            state = np.reshape(env.boards, (1,81))
            action = index_to_coordinates(agent.act(env, state))
            
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
    liste_gagne.append(gagne_dqn)
    liste_perdu.append(perdu_dqn)
        
    
    plt.bar(["gagné","nul","perdu"],[gagne_dqn,neutre_dqn,perdu_dqn],color = ['tab:green', 'tab:blue', 'tab:red'])
    plt.ylabel('Nombre de parties')
    plt.title('Parties jouées par un agent DQN contre un agent aléatoire')
    plt.show()
    
plt.plot(NB_EPISODES_TRAIN,liste_gagne,label="DQN gagné")
plt.plot(NB_EPISODES_TRAIN,liste_perdu,label="DQN perdu")
plt.legend()
plt.xlabel('Nombre de simulations')
plt.ylabel('Nombre de parties')
plt.title('Sur 100 parties jouées, nombre de parties gagnées et\nperdues en fonction du nombre d\'épisodes')
plt.show()
