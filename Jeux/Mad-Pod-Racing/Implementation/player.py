import math
import random
import pygame


class Player:
    def __init__(self, id, name, checkpoints, player_send_q, player_receive_q, screen=None, colour=None, verbose=False):
        self.id = id
        self.name = name
        self.screen = screen
        self.pos = pygame.Vector2(checkpoints[0])
        self.vel = pygame.Vector2(0, 0)
        self.angle = self.angle_to(checkpoints[1])
        self.next_checkpoint = 1
        self.checkpoints = checkpoints
        self.laps = 0
        self.player_send_q = player_send_q
        self.player_receive_q = player_receive_q
        self.boost_available = True
        self.colour = colour
        self.verbose = verbose

    def distance_to(self, point):
        # Computes the distance between the player and a given point
        return math.sqrt((self.pos.x - point.x)**2 + (self.pos.y - point.y)**2)

    def angle_to(self, point):
        # Computes the angle between the player and a given point
        d = self.distance_to(point)
        dx = (point.x - self.pos.x) / d
        dy = (point.y - self.pos.y) / d

        a = math.acos(dx) * 180 / math.pi

        if dy < 0:
            a = 360 - a

        return a

    def draw(self):
        # Displays the player on the screen
        pygame.draw.circle(self.screen, self.colour, self.pos, 20)
        pygame.draw.line(self.screen, "green", self.pos, self.pos + self.vel/2, 5)

    def update(self, dt, gui=False):
        # Get player input
        target_x, target_y, thrust = self.inquire_user()

        if self.verbose:
            print(f"{self.name} input: {target_x}, {target_y}, {thrust}")

        # Reorient the pod (max 18 degrees)
        a = self.angle_to(self.checkpoints[self.next_checkpoint])

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

        # Update player position and velocity

        ra = self.angle * math.pi / 180

        self.vel.x += math.cos(ra) * thrust
        self.vel.y += math.sin(ra) * thrust

        self.pos += self.vel * dt

        self.pos.x = round(self.pos.x)
        self.pos.y = round(self.pos.y)

        self.vel.x = int(self.vel.x * .85)
        self.vel.y = int(self.vel.y * .85)

        if self.verbose:
            print(f"{self.name} state: {self.pos=}, {self.vel=}, {self.angle=}")

        # Detect checkpoint reached
        if self.pos.distance_to(self.checkpoints[self.next_checkpoint]) < (80 if gui else 800):
            if self.next_checkpoint == 0:
                self.laps += 1
                if self.laps == 3:
                    if self.verbose:
                        print(f"Player {self.name} wins")
                    return self.id
            self.next_checkpoint = (self.next_checkpoint + 1) % len(self.checkpoints)

        return 0  # Continue game

    def inquire_user(self):
        # Compute next checkpoint distance and angle

        checkpoint = self.checkpoints[self.next_checkpoint]
        distance = round(self.pos.distance_to(checkpoint))
        angle = round(self.pos.angle_to(checkpoint))

        # Send data to player
        self.player_receive_q.put(f"{int(self.pos.x) * 10} {int(self.pos.y) * 10} {int(self.checkpoints[self.next_checkpoint].x) * 10} {int(self.checkpoints[self.next_checkpoint].y) * 10} {distance * 10} {angle}")

        target_x, target_y, thrust = self.player_send_q.get().split()
        target_x = int(target_x) // 10
        target_y = int(target_y) // 10
        if thrust == "BOOST":
            thrust = 1000 if self.boost_available else 100
            self.boost_available = False
        else:
            thrust = int(thrust)

        self.player_send_q.task_done()

        return target_x, target_y, thrust
