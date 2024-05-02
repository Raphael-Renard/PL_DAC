import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from Game_representation import Morpion, MorpionSimple



class PolicyGradient:
    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.99):
        self.policy_network = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            #nn.Linear(162, 512),
            #nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_size),
            nn.Softmax(dim=-1))
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.gamma = gamma

    def select_action(self, env, state):
        state_tensor = torch.FloatTensor(state)
        action_probs = self.policy_network(state_tensor)

        if torch.isnan(action_probs).any():
            print('action probs',action_probs)
            print('state tensor',state_tensor)
            print('state',state)

        
        ###
        # Get legal actions
        possible_moves = env.get_possible_moves()
        legal_actions = [coordinates_to_index(move) for move in possible_moves]

        
        # Mask action probabilities to only include legal actions
        masked_probs = torch.zeros_like(action_probs)
        masked_probs[0][legal_actions] = action_probs[0][legal_actions]

        masked_probs /= masked_probs.sum() # Normalize probabilities
        action_dist = torch.distributions.Categorical(masked_probs)

        ###
        
        #action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)

        return action.item(), log_prob

    def update(self, rewards, log_probs):
        discounted_rewards = self.calculate_discounted_rewards(rewards)
        policy_loss = []
        for log_prob, discounted_reward in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * discounted_reward)
        policy_loss = torch.cat(policy_loss).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()


    def calculate_discounted_rewards(self, rewards):
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(rewards):
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        if discounted_rewards.size(0)>1:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (torch.std(discounted_rewards, dim=0) + 1e-8)
        return discounted_rewards
    

    def train_agent(self, env, num_episodes = 1000,show_plot=False,verbose_gradient=False):
    
        train_loss = []

        for episode in range(num_episodes):
            env.reset()
            
            rewards = []
            log_probs = []
            done = False

            while not done:
                state = np.reshape(env.boards, (1, 81))
                action, log_prob = self.select_action(env,state)
                next_state, reward, done = env.step2(action)
                rewards.append(reward)
                log_probs.append(log_prob)

            self.update(rewards, log_probs)

            if verbose_gradient:
                if (episode + 1) % 10 == 0:
                    for name, param in self.policy_network.named_parameters():
                        if param.grad is not None:
                            print(f"Parameter: {name}, Gradient Norm: {param.grad.norm().item()}")
            

            episode_loss = -torch.cat(log_probs).sum().item() #
            train_loss.append(episode_loss)

            if (episode + 1) % 10 == 0:
                print(f"Episode: {episode + 1}, Total Reward: {sum(rewards)}, Loss: {episode_loss}")

        if show_plot:
            plt.plot(train_loss)
            plt.title("Loss pendant l'entraînement")
            plt.xlabel('Episodes')
            plt.ylabel('Loss')
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

NB_EPISODES = range(500,1000,50)


for nb_episodes in NB_EPISODES:
    gagne_pg = 0
    perdu_pg = 0
    neutre_pg = 0

    env = MorpionSimple()
    state_size = 81  
    action_size = 81 

    agent = PolicyGradient(state_size, action_size)
    agent.train_agent(env, nb_episodes, show_plot=True)

    for partie in range(100):
        env.reset()
        alea = Aleatoire(env)
        T = False

            
        while not T:
            state = np.reshape(env.boards, (1,81))
            action = index_to_coordinates(agent.select_action(env, state)[0])
            
            env.make_move_self(action)
            T = env.is_terminal((action[0]//3,action[1]//3))
            if T:
                break
            opponent_move = alea.give_move()
            env.make_move_self(opponent_move)
            T = env.is_terminal((opponent_move[0]//3,opponent_move[1]//3))

        resultat = env.get_result()
        if resultat == 1:
            gagne_pg +=1
        elif resultat == -1:
            perdu_pg +=1
        else:
            neutre_pg +=1
        
    liste_gagne.append(gagne_pg)
    liste_perdu.append(perdu_pg)

    """
    plt.bar(["gagné","nul","perdu"],[gagne_pg,neutre_pg,perdu_pg],color = ['tab:green', 'tab:blue', 'tab:red'])
    plt.ylabel('Nombre de parties')
    plt.title('Policy gradient contre aléatoire')
    plt.show()
    """


plt.plot(NB_EPISODES,liste_gagne,label="policy gradient gagné")
plt.plot(NB_EPISODES,liste_perdu,label="policy gradient perdu")
plt.legend()
plt.xlabel('Nombre de simulations')
plt.ylabel('Nombre de parties')
plt.title('Sur 100 parties jouées, nombre de parties gagnées et\nperdues en fonction du nombre d\'épisodes')
plt.show()
