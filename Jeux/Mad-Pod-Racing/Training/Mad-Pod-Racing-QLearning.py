import math
import pickle
import random
from Qlearning import q_learning#, Game
import numpy as np
import pygame
from Jeux.Qlearning import Game


# autres solutions faire varier la discretisation en fonction de la distance aux cheickpoints 
# utiliser dico a la place de table de hachage
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


def discretiser_angle(angle):
    # Limits to a 90° vision
    if angle < -45:
        angle = -45
    elif angle > 45:
        angle = 45

    # Discretize into 9 values ranging from -4 to 4
    angle //= 10
    if angle < 0:
        angle -= 1

    return angle


def discretiser_etat(checkpoint_pos, player_pos, angle, velocity):
    max_dist = 8000
    distance_intervals = np.exp(np.log(max_dist - 800) * np.arange(.1, 1.1, .1))

    distance_to_checkpoint = checkpoint_pos.distance_to(player_pos) - 800
    dcheckpoint_disc = 0
    while dcheckpoint_disc < 9 and distance_to_checkpoint < distance_intervals[dcheckpoint_disc]:
        dcheckpoint_disc += 1

    checkpoint_angle = angle_to(player_pos, checkpoint_pos)
    angle_to_checkpoint = diff_angle(angle, checkpoint_angle)

    acheckpoint_disc = discretiser_angle(angle_to_checkpoint)

    velocity_length = min(500, int(velocity.length()))
    velocity_length //= 100  # 500 values / 100 -> 5 values

    velocity_angle = angle_to(player_pos, player_pos + velocity)
    angle_to_velocity = diff_angle(angle, velocity_angle)
    avelocity_disc = discretiser_angle(angle_to_velocity)

    pol_next_checkpoint = (dcheckpoint_disc, acheckpoint_disc)  # 10*10 = 100 values
    pol_velocity = (velocity_length, avelocity_disc)  # 5*10 = 50 values

    return pol_next_checkpoint, pol_velocity  # 100 * 50 = 5000 values


class MadPodRacingQLearning(Game):
    def __init__(self):
        super().__init__()
        self.nb_actions = 50
        self.nb_states = 4050

        self.player_pos = pygame.Vector2(random.randint(1500, 14500), random.randint(1500, 7500))
        self.checkpoint_pos = pygame.Vector2(random.randint(1500, 14500), random.randint(1500, 7500))

        self.angle = random.randint(0, 359)
        self.velocity = pygame.Vector2(random.randint(0, 400), random.randint(0, 400))

        self.player_state = discretiser_etat(self.checkpoint_pos, self.player_pos, self.angle, self.velocity)

    def reset(self):
        self.player_pos = pygame.Vector2(random.randint(1500, 14500), random.randint(1500, 7500))
        self.checkpoint_pos = pygame.Vector2(random.randint(1500, 14500), random.randint(1500, 7500))

        self.angle = random.randint(0, 359)
        self.velocity = pygame.Vector2(random.randint(0, 400), random.randint(0, 400))

        self.player_state = discretiser_etat(self.checkpoint_pos, self.player_pos, self.angle, self.velocity)
        return self.player_state

    def step(self, action, verbose=False):
        angles = [-18, -11, -6, -3, -1, 1, 3, 6, 11, 18]

        thrust = (action // 10) * 25
        if thrust == 0:
            thrust = 1
        angle = (self.angle + angles[action % 10]) % 360

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

        # on utilise la distance entre le pod et le cheickpoint poour definir la récompense 
        if distance_to_checkpoint < 800:
            reward = 100
        else:
            reward = -distance_to_checkpoint / 100 + self.velocity.length() / 100

        self.player_state = discretiser_etat(self.checkpoint_pos, self.player_pos, self.angle, self.velocity)
        # fin 
        if distance_to_checkpoint < 800:
            done = True
        else:
            done = False

        return self.player_state, reward, done


new_qtable = True
nb_tests = 10000

env = MadPodRacingQLearning()

if new_qtable:
    q_table = q_learning(env, num_episodes=100000)

    # Dump q_table
    with open('q_table.pkl', 'wb') as f:
        pickle.dump(q_table, f)

else:
    # Load q_table
    with open('q_table.pkl', 'rb') as f:
        q_table = pickle.load(f)

# Test de l'agent entraîné
times = np.array([env.test(q_table, 100) for _ in range(nb_tests)])

print(f"Pourcentage de timeout : {round(len(times[times==-1]) / nb_tests, 4) * 100}\nPourcentage d'états non reconnus : {round(len(times[times==-2]) / nb_tests, 4) * 100}\nNombre d'itérations moyen : {times[times>=0].mean()}")
