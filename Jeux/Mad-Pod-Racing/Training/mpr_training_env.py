import functools
import math
import random

import numpy as np


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


def discretiser_angle(angle, nb_discretisations, max_angle=45):
    nb_discretisations += 1
    nb_par_cote = nb_discretisations // 2

    # On restreint à une vision devant soi
    if angle < -45:
        angle = -45
    elif angle > 45:
        angle = 45

    side_intervals = np.round(np.exp(np.log(max_angle) * np.arange(0, 1.1, 1 / nb_par_cote))[1:])

    if nb_discretisations % 2 == 1:
        angle_intervals = np.concatenate((-side_intervals[::-1], np.array([0]), side_intervals))[1:]
    else:
        angle_intervals = np.concatenate((-side_intervals[::-1], side_intervals))[1:]

    disc_angle = 0

    while angle > angle_intervals[disc_angle]:
        disc_angle += 1

    # print(angle, disc_angle, angle_intervals)

    return disc_angle


def discretiser_distance(distance, nb_discretisations, max_distance=8000, log=True):
    if log:
        distance_intervals = np.round(np.exp(np.log(max_distance - 800) * np.arange(0, 1.1, 1 / nb_discretisations))[1:])
    else:
        distance_intervals = np.linspace(0, max_distance, nb_discretisations + 1)[1:]

    # print(distance, distance_intervals)

    disc_dist = 0
    while distance - 800 > distance_intervals[disc_dist] and disc_dist < (nb_discretisations - 1):
        disc_dist += 1

    # print(distance, disc_dist, distance_intervals)

    return disc_dist


def discretiser_etat(checkpoint_pos, player_pos, angle, speed, discretisations=(5, 9, 5, 9)):
    dist_checkpoint = distance_to(player_pos, checkpoint_pos)
    disc_dist_checkpoint = discretiser_distance(dist_checkpoint, discretisations[0])

    checkpoint_angle = angle_to(player_pos, checkpoint_pos)
    angle_to_checkpoint = diff_angle(angle, checkpoint_angle)

    acheckpoint_disc = discretiser_angle(angle_to_checkpoint, discretisations[1])

    speed_length = distance_to((0, 0), speed)
    disc_speed_length = discretiser_distance(speed_length, discretisations[2], log=False, max_distance=500)

    speed_angle = angle_to(player_pos, player_pos + speed)
    angle_to_speed = diff_angle(angle, speed_angle)
    aspeed_disc = discretiser_angle(angle_to_speed, discretisations[3])

    pol_next_checkpoint = (disc_dist_checkpoint, acheckpoint_disc)
    pol_speed = (disc_speed_length, aspeed_disc)

    return pol_next_checkpoint, pol_speed


def make(discretisations_etat=None, discretisations_action=None):
    return Env(discretisations_etat, discretisations_action)


class Env:
    def __init__(self, discretisations_etat=None, discretisations_action=None):
        # None -> continu
        # Sinon tuple (#Distances checkpoint, #Angles checkpoint, #Longueurs vitesse, #Angles vitesse)
        self.discretisations_etat = discretisations_etat

        # None -> continu
        # Sinon tuple (#Angles (target), #Delta thrust)
        self.discretisations_action = discretisations_action

        if discretisations_etat:
            self.nb_states = functools.reduce(lambda x, y: x * y, discretisations_etat)
        if discretisations_action:
            self.nb_actions = functools.reduce(lambda x, y: x * y, discretisations_action)

        self.prev_thrust = None
        self.player_state = None
        self.speed = None
        self.angle = None
        self.player_pos = None
        self.checkpoint_pos = None
        self.iteration = None

        self.reset()

    def reset(self):
        self.checkpoint_pos = (random.randint(1500, 14500), random.randint(1500, 7500))
        self.player_pos = (random.randint(1500, 14500), random.randint(1500, 7500))

        while distance_to(self.checkpoint_pos, self.player_pos) < 800:
            self.player_pos = (random.randint(1500, 14500), random.randint(1500, 7500))

        self.angle = random.randint(0, 359)
        self.speed = (random.randint(0, 400), random.randint(0, 400))

        self.player_state = discretiser_etat(self.checkpoint_pos, self.player_pos, self.angle, self.speed)

        self.prev_thrust = 100
        self.iteration = 0

        return self.player_state

    def step(self, action, verbose=False):
        target_x, target_y, thrust = self._unpack_action(action)
        # print(target_x, target_y, thrust, distance_to(self.player_pos, self.checkpoint_pos))

        self.iteration += 1

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

        self.speed = (math.cos(ra) * thrust, math.sin(ra) * thrust)

        if verbose:
            print("player_pos before step: ", self.player_pos)

        self.player_pos = ((self.player_pos[0] + self.speed[0]) * dt, (self.player_pos[1] + self.speed[1]) * dt)

        self.player_pos = (round(self.player_pos[0]), round(self.player_pos[1]))

        if verbose:
            print("player_pos after step: ", self.player_pos)

        self.speed = (int(self.speed[0] * .85), int(self.speed[1] * .85))

        reward = self._get_reward()

        distance_to_checkpoint = distance_to(self.player_pos, self.checkpoint_pos)

        if self.discretisations_etat:
            self.player_state = discretiser_etat(self.checkpoint_pos, self.player_pos, self.angle, self.speed)
        else:
            self.player_state = self.checkpoint_pos, self.player_pos, self.angle, self.speed

        if distance_to_checkpoint < 800 or self.iteration == 100:
            done = True
        else:
            done = False

        return self.player_state, reward, done

    def test(self, q_table, timeout):
        state = self.reset()
        done = False
        time = 0

        while not done:
            time += 1
            if time == timeout:
                return -1

            try:
                action = np.argmax(q_table[state])
            except KeyError:
                print(state)
                return -2
            state, reward, done = self.step(action)

        return time

    def _unpack_action(self, action: int):
        """
        Dé-discrétise l'action
        :param action: entier représentant l'action discrétisée
        :return: target_x, target_y et thrust
        """
        nb_par_cote = self.discretisations_action[0] // 2
        side_intervals = np.round(np.exp(np.log(18) * np.arange(0, 1.1, 1 / nb_par_cote))[1:])
        angles = np.concatenate((-side_intervals[::-1], np.array([0]) if self.discretisations_action[0] % 2 == 1 else np.array(None), side_intervals))

        dthrusts = np.linspace(-50, 50, self.discretisations_action[1])

        # print(f"{self.nb_actions=}, {self.discretisations_action=}, {action=}")

        thrust = self.prev_thrust + dthrusts[action // self.discretisations_action[0]]
        thrust = max(0, min(100, thrust))

        self.prev_thrust = thrust

        angle = (self.angle + angles[action % len(angles)]) % 360

        target_x = self.player_pos[0] + 10000 * math.cos(math.radians(angle))
        target_y = self.player_pos[1] + 10000 * math.sin(math.radians(angle))

        return round(target_x), round(target_y), thrust

    def _get_reward(self):
        distance_to_checkpoint = distance_to(self.player_pos, self.checkpoint_pos)

        # on utilise la distance entre le pod et le checkpoint pour définir la récompense
        if distance_to_checkpoint < 800:
            return 100
        else:
            return -distance_to_checkpoint/100#np.exp(-distance_to_checkpoint)
