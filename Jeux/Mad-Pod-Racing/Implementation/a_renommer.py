import random
import sys
import threading
from threading import Event

import pygame


def generate_checkpoints():
    checkpoints = []
    for _ in range(random.randint(4, 6)):
        checkpoint = pygame.Vector2(random.randint(150, 1450), random.randint(150, 750))
        while any(checkpoint.distance_to(checkpoint2) < 500 for checkpoint2 in checkpoints):
            checkpoint = pygame.Vector2(random.randint(100, 1500), random.randint(100, 800))
        checkpoints.append(checkpoint)

    return checkpoints


class Player:
    def __init__(self, screen, pos, checkpoints, player_send_q, player_receive_q):
        self.screen = screen
        self.pos = pos
        self.vel = pygame.Vector2(0, 0)
        self.acc = pygame.Vector2(0, 0)
        self.next_checkpoint = 1
        self.checkpoints = checkpoints
        self.player_send_q = player_send_q
        self.player_receive_q = player_receive_q

    def draw(self):
        pygame.draw.circle(self.screen, "white", self.pos, 20)

    def update(self, dt):
        eps = 0.0001

        target_x, target_y, thrust = self.inquire_user()

        print(f"Player input: {target_x}, {target_y}, {thrust}")

        # self.acc = (pygame.Vector2(target_x, target_y) - self.pos).normalize() * (50 - thrust) * 3
        # self.vel += self.acc * dt
        self.vel = (pygame.Vector2(target_x, target_y) - self.pos).normalize() * thrust * 3
        self.pos += self.vel * dt
        print(f"Player state: {self.pos}, {self.vel}, {self.acc}")

        if self.pos.distance_to(self.checkpoints[self.next_checkpoint]) < 80:
            self.next_checkpoint = (self.next_checkpoint + 1) % len(self.checkpoints)

    def inquire_user(self):
        # Compute next checkpoint distance and angle
        checkpoint = self.checkpoints[self.next_checkpoint]
        distance = round(self.pos.distance_to(checkpoint))
        angle = round(self.pos.angle_to(checkpoint))

        # User will input the target x, y and thrust by printing them to the console, using values sent by the server
        # Check if the line has been printed to the console
        self.player_receive_q.put(f"{int(self.pos.x)} {int(self.pos.y)} {int(self.checkpoints[self.next_checkpoint].x)} {int(self.checkpoints[self.next_checkpoint].y)} {distance} {angle}")

        # print("Server : input sent", file=sys.stderr, flush=True)

        target_x, target_y, thrust = map(int, self.player_send_q.get().split())
        self.player_send_q.task_done()

        # print("Server : input received", file=sys.stderr, flush=True)

        # Put the process back to sleep
        return target_x, target_y, thrust


def player_loop(player_send_q, player_receive_q):
    t = 0
    x, y = 0, 0
    # game loop
    boost = True
    while True:
        try:
            t += 1
            # next_checkpoint_x: x position of the next check point
            # next_checkpoint_y: y position of the next check point
            # next_checkpoint_dist: distance to the next checkpoint
            # next_checkpoint_angle: angle between your pod orientation and the direction of the next checkpoint
            ax, ay = x, y

            x, y, next_checkpoint_x, next_checkpoint_y, next_checkpoint_dist, next_checkpoint_angle = [int(i) for i in player_receive_q.get().split()]
            player_receive_q.task_done()

            # print("Client : input received", file=sys.stderr, flush=True)

            dx, dy = abs(ax - x), abs(ay - y)

            #opponent_x, opponent_y = [int(i) for i in input().split()]

            target_x = next_checkpoint_x
            target_y = next_checkpoint_y

            # Write an action using print
            # To debug: print("Debug messages...", file=sys.stderr, flush=True)

            # You have to output the target position
            # followed by the power (0 <= thrust <= 100)
            # i.e.: "x y thrust"
            thrust = 100

            if abs(next_checkpoint_angle) > 45:
                thrust -= abs(next_checkpoint_angle) - 20
            else:
                if next_checkpoint_dist < 3000:
                    thrust -= int((3000 - next_checkpoint_dist) / 30)

                # thrust = max(thrust, 50)

            thrust = max(thrust, 0)

            player_send_q.put(f"{target_x} {target_y} {thrust}")

            # print("Client : input sent", file=sys.stderr, flush=True)

        except Exception as e:
            print(f"Error: {e}")
            break
