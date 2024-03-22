import random
import gym
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt



class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95   
        self.epsilon = 1.0  
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()  
        self.target_model = self._build_model()  

    def _build_model(self):
        model = Sequential()
        model.add(Dense(50, input_dim=self.state_size, activation='relu'))
        #model.add(Dense(512, input_dim=self.state_size, activation='relu'))
        model.add(Dense(100, activation='relu'))
        #model.add(Dense(256, activation='relu'))
        #model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values)
        
    
    def replay(self,batch_size):

        minibatch = random.sample(self.memory, batch_size)
        state = np.zeros((batch_size, self.state_size))
        next_state = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []
        for i in range(batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        target = self.model.predict(state)
        target_next = self.target_model.predict(next_state)

        for i in range(batch_size):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        self.model.fit(state, target,batch_size, verbose=0)
        if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay


        """def replay(self, batch_size):
            minibatch = random.sample(self.memory, batch_size)
            for state, action, reward, next_state, done in minibatch:
                target = reward
                if not done:
                    target = reward + self.gamma * np.max(self.target_model.predict(next_state))
                target_f = self.model.predict(state)
                target_f[0][action] = target
                self.model.fit(state, target_f,batch_size, epochs=1, verbose=1)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay"""

    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())


def plot_res(values):   
    plt.plot(values, label='score per run')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    x = range(len(values))
    plt.legend()
    # Calculate the trend
    try:
        z = np.polyfit(x, values, 1)
        p = np.poly1d(z)
        plt.plot(x,p(x),"--", label='trend')
    except:
        print('')

    plt.show()



# Initialisation de l'environnement Gym
        

env = gym.make('CartPole-v1',render_mode = "human")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = DQNAgent(state_size, action_size)
batch_size = 64
n_episodes = 50
C = 10
final = []

for episode in range(n_episodes):
    state = env.reset()[0]  
    state = np.reshape(state, [1, state_size])

    total_reward = 0
    done = False

    while not done:
        #env.render()
        action = agent.act(state)
        next_state, reward, done, _,_ = env.step(action)

        next_state = np.reshape(next_state, [1, state_size])
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
    #if(total_reward > 100): 
        #break

    final.append(total_reward)
plot_res(final)

