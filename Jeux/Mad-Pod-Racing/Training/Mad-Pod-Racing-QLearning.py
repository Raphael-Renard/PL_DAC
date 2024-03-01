import math
import pickle
import random
from Qlearning import q_learning#, Game
import numpy as np
import pygame
from Jeux.Qlearning import Game


# class MadPodRacingQLearning(Game):
#     def __init__(self):
#         super().__init__()
#         self.nb_actions = 50
#         self.nb_states = 25000
#         self.player_pos = pygame.Vector2(random.randint(150, 1450), random.randint(150, 750))
#         self.checkpoint_pos = pygame.Vector2(random.randint(150, 1450), random.randint(150, 750))
#         self.velocity = pygame.Vector2(random.randint(0, 360), random.randint(0, 360))
#         self.acceleration = pygame.Vector2(random.randint(-360, 360), random.randint(-360, 360))
#         self.player_state = self.discretiser_etat(self.player_pos, self.velocity, self.acceleration, self.checkpoint_pos)
#
#     def discretiser_etat(self, pos: pygame.Vector2, velocity: pygame.Vector2, acceleration: pygame.Vector2, next_checkpoint: pygame.Vector2):
#         pol_next_checkpoint = next_checkpoint.distance_to(pos) // 1800, next_checkpoint.angle_to(velocity) // 36
#         pol_velocity = velocity.length() // 50, 0
#         pol_acceleration = acceleration.length() // 100, acceleration.angle_to(velocity) // 72
#         return pol_next_checkpoint, pol_velocity, pol_acceleration
#
#     def reset(self):
#         player_pos = pygame.Vector2(random.randint(1500, 14500), random.randint(1500, 7500))
#         checkpoint_pos = pygame.Vector2(random.randint(1500, 14500), random.randint(1500, 7500))
#         velocity = pygame.Vector2(random.randint(0, 3600), random.randint(0, 3600))
#         acceleration = pygame.Vector2(random.randint(-3600, 3600), random.randint(-3600, 3600))
#         self.player_state = self.discretiser_etat(player_pos, velocity, acceleration, checkpoint_pos)
#
#     def step(self, action):
#         thrust = (action // 10) * 25
#         angle = (action % 10) * 36
#
#         target_x = thrust * math.cos(math.radians(angle))
#         target_y = thrust * math.sin(math.radians(angle))
#
#         dt = 1/60
#
#         self.acceleration = .984 * self.acceleration + (pygame.Vector2(target_x, target_y) - self.pos).normalize() * thrust
#         self.velocity += self.acceleration * dt
#
#         max_speed = 500
#
#         # Limit speed
#         if self.velocity.length() > max_speed:
#             self.velocity.scale_to_length(max_speed)
#
#         # Update position
#         self.player_pos += self.velocity * dt + 0.5 * self.acceleration * dt**2
#
#         self.player_state = self.discretiser_etat(self.player_pos, self.velocity, self.acceleration, self.checkpoint_pos)
#
#         if self.player_pos.distance_to(self.checkpoint_pos) < 800:
#             done = True
#         else:
#             done = False
#
#         return self.player_state, -1, done


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

    a = math.acos(dx) * 180 / math.pi

    if dy < 0:
        a = 360 - a

    return a


def diff_angle(a1, a2):
    right = a2 - a1 if a1 <= a1 else 360 - a1 + a2
    left = a1 - a2 if a1 >= a2 else a1 + 360 - a2

    return right if right < left else -left


class MadPodRacingQLearning(Game):
    def __init__(self):
        super().__init__()
        self.nb_actions = 50
        self.nb_states = 5000

        self.player_pos = pygame.Vector2(random.randint(1500, 14500), random.randint(1500, 7500))
        self.checkpoint_pos = pygame.Vector2(random.randint(1500, 14500), random.randint(1500, 7500))

        self.angle = random.randint(0, 359)
        self.velocity = pygame.Vector2(random.randint(0, 400), random.randint(0, 400))

        self.player_state = self.discretiser_etat()

    def discretiser_etat(self):
        distance_to_checkpoint = min(8000, int(self.checkpoint_pos.distance_to(self.player_pos)))
        distance_to_checkpoint //= 800  # 8000 values / 800 -> 10 values

        checkpoint_angle = angle_to(self.player_pos, self.checkpoint_pos)
        angle_to_checkpoint = diff_angle(self.angle, checkpoint_angle) % 360
        angle_to_checkpoint //= 36  # 360 values / 36 -> 10 values

        velocity_length = min(500, int(self.velocity.length()))
        velocity_length //= 100  # 500 values / 100 -> 5 values

        velocity_angle = angle_to(self.player_pos, self.player_pos + self.velocity)
        angle_to_velocity = diff_angle(self.angle, velocity_angle) % 360
        angle_to_velocity //= 36  # 360 values / 36 -> 10 values

        pol_next_checkpoint = (int(distance_to_checkpoint / 5), int(angle_to_checkpoint / 5))  # 10*10 = 100 values
        pol_velocity = (velocity_length, angle_to_velocity)  # 5*10 = 50 values

        return pol_next_checkpoint, pol_velocity  # 100 * 50 = 5000 values

    def reset(self):
        self.player_pos = pygame.Vector2(random.randint(1500, 14500), random.randint(1500, 7500))
        self.checkpoint_pos = pygame.Vector2(random.randint(1500, 14500), random.randint(1500, 7500))

        self.angle = random.randint(0, 359)
        self.velocity = pygame.Vector2(random.randint(0, 400), random.randint(0, 400))

        self.player_state = self.discretiser_etat()
        return self.player_state

    def step(self, action, verbose=False):

        thrust = (action // 10) * 25
        angle = (self.angle + (action % 10) * 36) % 360

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
            reward = -distance_to_checkpoint / 100

        self.player_state = self.discretiser_etat()
        # fin 
        if distance_to_checkpoint < 800:
            done = True
        else:
            done = False

        return self.player_state, reward, done


env = MadPodRacingQLearning()
q_table = q_learning(env)

# Dump q_table
with open('q_table.pkl', 'wb') as f:
    pickle.dump(q_table, f)

# Pour juste tester la qtable enregistrée décommenter le load et commenter le qlearning et dump

# Load q_table
# with open('q_table.pkl', 'rb') as f:
#     q_table = pickle.load(f)

# Test de l'agent entraîné
state = env.reset()
done = False
while not done:
    action = np.argmax(q_table[state])
    print(action)
    state, reward, done = env.step(action, verbose=True)
