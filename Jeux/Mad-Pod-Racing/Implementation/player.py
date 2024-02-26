import random
import pygame


class Player:
    def __init__(self, name, checkpoints, player_send_q, player_receive_q, screen=None, colour=None, verbose=False):
        self.name = name
        self.screen = screen
        self.pos = pygame.Vector2(checkpoints[0])
        self.vel = pygame.Vector2(0, 0)
        self.acc = pygame.Vector2(0, 0)
        self.next_checkpoint = 1
        self.checkpoints = checkpoints
        self.laps = 0
        self.player_send_q = player_send_q
        self.player_receive_q = player_receive_q
        self.boost_available = True
        self.colour = colour
        self.verbose = verbose

    def draw(self):
        pygame.draw.circle(self.screen, self.colour, self.pos, 20)
        pygame.draw.line(self.screen, "green", self.pos, self.pos + self.vel, 5)

    def update(self, dt):
        max_speed = 150

        # Get player input
        target_x, target_y, thrust = self.inquire_user()

        if self.verbose:
            print(f"{self.name} input: {target_x}, {target_y}, {thrust}")

        # Compute new acceleration / velocity
        self.acc = .95 * self.acc + (pygame.Vector2(target_x, target_y) - self.pos).normalize() * thrust
        self.vel += self.acc * dt

        # Limit speed
        if self.vel.length() > max_speed:
            self.vel.scale_to_length(max_speed)

        # Update position
        self.pos += self.vel * dt + 0.5 * self.acc * dt**2

        if self.verbose:
            print(f"{self.name} state: {self.pos}, {self.vel}, {self.acc}")

        # Detect checkpoint reached
        if self.pos.distance_to(self.checkpoints[self.next_checkpoint]) < 80:
            if self.next_checkpoint == 0:
                self.laps += 1
                if self.laps == 3:
                    if self.verbose:
                        print(f"Player {self.name} wins")
                    return True  # End game
            self.next_checkpoint = (self.next_checkpoint + 1) % len(self.checkpoints)

        return False  # Continue game

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
