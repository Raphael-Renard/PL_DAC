import math
import random
import numpy as np
import pygame
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

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


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0  # on commmence par explorer puis on réduit l'exploration
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()  # Model principal Q
        self.target_model = self._build_model()  # Modèle cible Q_target

    def _build_model(self):
        # nombre de couches ? fonction d'activation ?
        """
        model = models.Sequential([
                layers.Input(shape=(self.state_size,)),
                # arbitraire
                layers.Dense(24, activation='relu'),
                layers.Dense(24, activation='relu'),
                layers.Dense(self.action_size, activation='linear')
        ])
        """
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')  # optimiseur sgd ?
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values)

    def learning(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states = np.zeros((batch_size, self.state_size)) #
        target_f = self.model.predict(states) #
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i]=state #
            target = reward
            if not done:
                target = reward + self.gamma * np.max(self.target_model.predict(next_state))
            #target_f = self.model.predict(state)
            target_f[i][action] = target
        self.model.fit(states, target_f, epochs=1, verbose=1)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())


class MadPodRacingDQN:
    def __init__(self):
        self.nb_actions = 9
        self.nb_states = 4
        self.player_pos = pygame.Vector2(random.uniform(1500, 14500), random.uniform(1500, 7500))
        self.checkpoint_pos = pygame.Vector2(random.uniform(1500, 14500), random.uniform(1500, 7500))
        self.angle = random.randint(0, 359)
        self.velocity = pygame.Vector2(random.uniform(0, 400), random.uniform(0, 400))
        self.prev_thrust = 0
        self.player_state = np.array([self.player_pos.x, self.player_pos.y, self.checkpoint_pos.x, self.checkpoint_pos.y])

    def reset(self):
        self.checkpoint_pos = pygame.Vector2(random.randint(1500, 14500), random.randint(1500, 7500))
        self.player_pos = pygame.Vector2(random.randint(1500, 14500), random.randint(1500, 7500))
        while self.player_pos.distance_to(self.checkpoint_pos) < 800:
            self.player_pos = pygame.Vector2(random.randint(1500, 14500), random.randint(1500, 7500))
        self.angle = random.randint(0, 359)
        self.velocity = pygame.Vector2(random.randint(0, 400), random.randint(0, 400))
        self.player_state = np.array([self.player_pos.x, self.player_pos.y, self.checkpoint_pos.x, self.checkpoint_pos.y])
        return self.player_state

    def get_state(self):
        distance_to_checkpoint = self.checkpoint_pos.distance_to(self.player_pos)

        checkpoint_angle = angle_to(self.player_pos, self.checkpoint_pos)
        angle_to_checkpoint = diff_angle(self.angle, checkpoint_angle)

        velocity_length = int(self.velocity.length())

        velocity_angle = angle_to(self.player_pos, self.player_pos + self.velocity)
        angle_to_velocity = diff_angle(self.angle, velocity_angle)

        return distance_to_checkpoint, angle_to_checkpoint, velocity_length, angle_to_velocity

    def step(self, action, verbose=False):
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
        return next_state, reward, done


env = MadPodRacingDQN()

state_size = env.nb_states
action_size = env.nb_actions
batch_size = 32
n_episodes = 1000
C = 10  # Fréquence de mise à jour du réseau cible

# Initialisation de l'agent DQN
agent = DQNAgent(state_size, action_size)

# Entraînement de l'agent
for episode in range(n_episodes):
    print("Episode : ", episode)
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            agent.update_target_model() #
            print("Episode:", episode + 1, ", Total Reward:", total_reward)
            break
        if len(agent.memory) > batch_size:
            agent.learning(batch_size)
            if episode % C == 0:
                agent.update_target_model()
