import codecs
import math
import pickle
import gzip
import numpy as np
import pygame
import sys
import os
import random
import gym
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import base64
import matplotlib.pyplot as plt

encoded_weight = """

"""

def _build_model(input_shape, action_space):
        model = nn.Sequential(
            nn.Linear(input_shape, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, action_space)
        )
        return model


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
    if angle < -max_angle:
        angle = -max_angle
    elif angle > max_angle:
        angle = max_angle

    side_intervals = np.round(np.exp(np.log(max_angle) * np.arange(0, 1.1, 1 / nb_par_cote))[1:])

    if nb_discretisations % 2 == 1:
        angle_intervals = np.concatenate((-side_intervals[::-1], np.array([0]), side_intervals))[1:]
    else:
        angle_intervals = np.concatenate((-side_intervals[::-1], side_intervals))[1:]

    disc_angle = np.digitize(angle, angle_intervals)
    # print(angle, disc_angle, angle_intervals)

    return disc_angle


def discretiser_distance(distance, nb_discretisations, min_distance=800, max_distance=10000, log=True):
    if log:
        distance_intervals = np.round(np.exp(np.log(max_distance - min_distance) * np.arange(0, 1.1, 1 / nb_discretisations))[1:])
    else:
        distance_intervals = np.linspace(min_distance, max_distance, nb_discretisations + 1)[1:]

    # print(distance, distance_intervals)

    disc_dist = np.digitize(distance, distance_intervals)

    # print(distance, disc_dist, distance_intervals)

    return disc_dist


def get_state(checkpoint_pos, player_pos, angle, speed, thrust_relatif=False, prev_thrust=0):
    dist_checkpoint = distance_to(player_pos, checkpoint_pos)

    checkpoint_angle = angle_to(player_pos, checkpoint_pos)
    angle_to_checkpoint = diff_angle(angle, checkpoint_angle)

    speed_length = distance_to((0, 0), speed)

    speed_angle = angle_to(player_pos, player_pos + speed)
    angle_to_speed = diff_angle(angle, speed_angle)

    if thrust_relatif:
        return (dist_checkpoint, angle_to_checkpoint, speed_length, angle_to_speed, prev_thrust)
    else:
        return (dist_checkpoint, angle_to_checkpoint, speed_length, angle_to_speed)


def discretiser_etat(checkpoint_pos, player_pos, angle, speed, discretisations, thrust_relatif=False, prev_thrust=0):
    player_state = get_state(checkpoint_pos, player_pos, angle, speed, discretisations, thrust_relatif, prev_thrust)

    # print(f"State before discretisation: {player_state}")
    if thrust_relatif:
        dist_checkpoint, angle_to_checkpoint, speed_length, angle_to_speed, prev_thrust = player_state
    else:
        dist_checkpoint, angle_to_checkpoint, speed_length, angle_to_speed = player_state

    disc_dist_checkpoint = discretiser_distance(dist_checkpoint, discretisations[0])
    disc_angle_checkpoint = discretiser_angle(angle_to_checkpoint, discretisations[1])
    disc_speed_length = discretiser_distance(speed_length, discretisations[2], log=False, min_distance=0, max_distance=1000)
    disc_angle_speed = discretiser_angle(angle_to_speed, discretisations[3])

    if thrust_relatif:
        disc_prev_thrust = discretiser_distance(prev_thrust, discretisations[4], min_distance=0, max_distance=100, log=False)
        return (disc_dist_checkpoint, disc_angle_checkpoint, disc_speed_length, disc_angle_speed, disc_prev_thrust)
    else:
        return (disc_dist_checkpoint, disc_angle_checkpoint, disc_speed_length, disc_angle_speed)


def unpack_action(action, player_pos, angle, discretisations_action, thrust_relatif=False, prev_thrust=0):
    """
    Dé-discrétise l'action
    :param action: entier représentant l'action discrétisée
    :return: target_x, target_y et thrust
    """
    nb_par_cote = discretisations_action[0] // 2
    side_intervals = np.round(np.exp(np.log(18) * np.arange(0, 1.1, 1 / nb_par_cote))[1:])
    angles = np.concatenate((-side_intervals[::-1], np.array([0]) if discretisations_action[0] % 2 == 1 else np.array(None), side_intervals))

    if thrust_relatif:
        dthrusts = np.round(np.linspace(-50, 50, discretisations_action[1]))

        # print(f"{nb_actions=}, {discretisations_action=}, {action=}")

        thrust = prev_thrust + dthrusts[action // discretisations_action[0]]
        thrust = max(0, min(100, thrust))

        prev_thrust = thrust

    else:
        thrusts = np.round(np.linspace(0, 100, discretisations_action[1]))
        thrust = thrusts[action // discretisations_action[0]]

    dangle = angles[action % len(angles)]

    angle = (angle + dangle) % 360

    target_x = player_pos[0] + 10000 * math.cos(math.radians(angle))
    target_y = player_pos[1] + 10000 * math.sin(math.radians(angle))

    return round(target_x), round(target_y), thrust, prev_thrust

#decompression
decoded_compressed_weights = base64.b64decode(encoded_weight)
decompressed_weights = gzip.decompress(decoded_compressed_weights)
model_weights = pickle.loads(decompressed_weights)

model = _build_model(5,9)
model.load_state_dict(model_weights)
model.eval()


t = 0
a, b = 0, 0
x, y = 0, 0

discretisations_etat, discretisations_action = None, (3, 3)
thrust_relatif = True

while True:

    t += 1

    ax, ay = x, y

    x, y, next_checkpoint_x, next_checkpoint_y, next_checkpoint_dist, next_checkpoint_angle = [int(i) for i in input().split()]
    input()

    if angle is None:
        angle = angle_to((x, y), (next_checkpoint_x, next_checkpoint_y))
    else:
        if next_checkpoint_angle > 18:
            next_checkpoint_angle = 18
        elif next_checkpoint_angle < -18:
            next_checkpoint_angle = -18

        angle += next_checkpoint_angle

    state = get_state()
    state_tensor = torch.tensor([state], dtype=torch.float32)
    
    with torch.no_grad():
        action_values = model(state_tensor)
    action = torch.argmax(action_values).item()