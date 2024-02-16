import math
import random

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



