import math
import random
from Qlearning import q_learning, Game
import numpy as np
import pygame
from Jeux.Qlearning import Game


class MadPodRacingQLearning(Game):
    def __init__(self):
        super().__init__()
        self.nb_actions = 50
        self.nb_states = 25000
        self.player_pos = pygame.Vector2(random.randint(150, 1450), random.randint(150, 750))
        self.checkpoint_pos = pygame.Vector2(random.randint(150, 1450), random.randint(150, 750))
        self.velocity = pygame.Vector2(random.randint(0, 360), random.randint(0, 360))
        self.acceleration = pygame.Vector2(random.randint(-360, 360), random.randint(-360, 360))
        self.player_state = self.discretiser_etat(self.player_pos, self.velocity, self.acceleration, self.checkpoint_pos)

    def discretiser_etat(self, pos: pygame.Vector2, velocity: pygame.Vector2, acceleration: pygame.Vector2, next_checkpoint: pygame.Vector2):
        pol_next_checkpoint = next_checkpoint.distance_to(pos) // 1800, next_checkpoint.angle_to(velocity) // 36
        pol_velocity = velocity.length() // 50, 0
        pol_acceleration = acceleration.length() // 100, acceleration.angle_to(velocity) // 72
        return pol_next_checkpoint, pol_velocity, pol_acceleration

    def reset(self):
        player_pos = pygame.Vector2(random.randint(1500, 14500), random.randint(1500, 7500))
        checkpoint_pos = pygame.Vector2(random.randint(1500, 14500), random.randint(1500, 7500))
        velocity = pygame.Vector2(random.randint(0, 3600), random.randint(0, 3600))
        acceleration = pygame.Vector2(random.randint(-3600, 3600), random.randint(-3600, 3600))
        self.player_state = self.discretiser_etat(player_pos, velocity, acceleration, checkpoint_pos)

    def step(self, action):
        thrust = (action // 10) * 25
        angle = (action % 10) * 36

        target_x = thrust * math.cos(math.radians(angle))
        target_y = thrust * math.sin(math.radians(angle))

        dt = 1/60

        self.acceleration = .984 * self.acceleration + (pygame.Vector2(target_x, target_y) - self.pos).normalize() * thrust
        self.velocity += self.acceleration * dt

        max_speed = 500

        # Limit speed
        if self.velocity.length() > max_speed:
            self.velocity.scale_to_length(max_speed)

        # Update position
        self.player_pos += self.velocity * dt + 0.5 * self.acceleration * dt**2

        self.player_state = self.discretiser_etat(self.player_pos, self.velocity, self.acceleration, self.checkpoint_pos)

        if self.player_pos.distance_to(self.checkpoint_pos) < 800:
            done = True
        else:
            done = False

        return self.player_state, -1, done


# autres solutions faire varier la discretisation en fonction de la distance aux cheickpoints 
# utiliser dico a la place de table de hachage

class MadPodRacingQLearning(Game):
    def __init__(self):
        super().__init__()
        self.nb_actions = 50
        self.nb_states = 8000
        self.player_pos = pygame.Vector2(random.randint(150, 1450), random.randint(150, 750))
        self.checkpoint_pos = pygame.Vector2(random.randint(150, 1450), random.randint(150, 750))

        self.velocity = pygame.Vector2(random.randint(0, 360), random.randint(0, 360))
        self.acceleration = pygame.Vector2(random.randint(-360, 360), random.randint(-360, 360))
        self.player_state = self.discretiser_etat(self.player_pos, self.velocity, self.acceleration, self.checkpoint_pos)
    
    def discretiser_etat(self, pos: pygame.Vector2, velocity: pygame.Vector2, acceleration: pygame.Vector2, next_checkpoint: pygame.Vector2):
        # etats possibles 20*20*20 = 8000
        distance_to_checkpoint = next_checkpoint.distance_to(pos)
        angle_to_checkpoint = next_checkpoint.angle_to(velocity)
        pol_next_checkpoint = (int(distance_to_checkpoint / 5), int(angle_to_checkpoint / 5))
        pol_velocity = (int(velocity.x / 5), int(velocity.y / 5))
        pol_acceleration = (int(acceleration.x / 5), int(acceleration.y / 5))
        return pol_next_checkpoint, pol_velocity, pol_acceleration
    
    def reset(self):
        self.player_pos = pygame.Vector2(random.randint(150, 1450), random.randint(150, 750))
        self.checkpoint_pos = pygame.Vector2(random.randint(150, 1450), random.randint(150, 750))
        self.velocity = pygame.Vector2(random.randint(0, 360), random.randint(0, 360))
        self.acceleration = pygame.Vector2(random.randint(-360, 360), random.randint(-360, 360))
        self.player_state = self.discretiser_etat(self.player_pos, self.velocity, self.acceleration, self.checkpoint_pos)
        return self.player_state
        
  
    def step(self, action):

        thrust = (action // 10) * 25
        angle = (action % 10) * 36

        target_x = thrust * math.cos(math.radians(angle))
        target_y = thrust * math.sin(math.radians(angle))

        dt = 1 / 60
        self.acceleration = .984 * self.acceleration + (pygame.Vector2(target_x, target_y) - self.player_pos).normalize() * thrust
        self.velocity += self.acceleration * dt

        # Limiter la vitesse maximale
        max_speed = 500
        if self.velocity.length() > max_speed:
            self.velocity.scale_to_length(max_speed)

        print("player_pos before step: ",self.player_pos)
        self.player_pos += self.velocity * dt + 0.5 * self.acceleration * dt ** 2
        print("player_pos after step: ", self.player_pos)


        distance_to_checkpoint = self.player_pos.distance_to(self.checkpoint_pos)

        print("distance_to_checkpoint :",distance_to_checkpoint)


        # on utilise la distance entre le pod et le cheickpoint poour definir la récompense 
        if distance_to_checkpoint < 800:
            reward = 100
        else:
            reward = -distance_to_checkpoint / 100

        self.player_state = self.discretiser_etat(self.player_pos, self.velocity, self.acceleration, self.checkpoint_pos)
        # fin 
        if distance_to_checkpoint < 50:
            done = True
        else:
            done = False
        return self.player_state, reward, done


env = MadPodRacingQLearning()
q_table = q_learning(env)

# Test de l'agent entraîné
state = env.reset()
done = False   
while not done:
    state_str = ','.join(map(str, state))
    action = np.argmax(q_table[state_str])
    state, reward, done = env.step(action)



