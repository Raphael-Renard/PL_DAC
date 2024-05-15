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
import traceback
import zlib
import base64 as b64

ascii_weights= """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 """

poids = pickle.loads(zlib.decompress(b64.b64decode(ascii_weights)))

class Neural_Network_Git():
    def __init__(self, input, output, hidden_layer_sizes=[], weights_biases=None):
        self.input = input # number of input nodes + 1 for bias
        self.output = output # number of output nodes

        # Layers
        self.layer_sizes = [self.input] + list(hidden_layer_sizes) + [self.output]
        self.layers = [np.ones(s) for s in self.layer_sizes]
        self.num_layers = len(self.layers) 

        # Weights and biases
        if weights_biases == None:
            self.weights = []
            self.biases = []
            for s0, s1 in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
                self.weights.append(np.random.randn(s0, s1))
                self.biases.append(np.random.randn(s1))
        else:
            self.weights, self.biases = weights_biases

        # Lists for enumeration
        self.weights_and_biases = list(zip(range(self.num_layers - 1), self.weights, self.biases))
        self.rev_weights_biases = list(reversed(self.weights_and_biases))


    def compute_output(self, inputs):
        # set inputs
        a = inputs
        self.layers[0] = a 

        for i, w, b in self.weights_and_biases:
            sum = np.dot(a,w.T) + np.array(b)
            a = 1 / (1 + np.exp(-sum))
            self.layers[i+1] = a
        return self.layers[-1]
    
    """def compute_output(self, inputs):
        a = inputs
        self.layers[0] = a

        for i, w, b in self.weights_and_biases:
            sum = np.dot(a, w.T) + b
            a = np.maximum(0, sum)# Utilisation de ReLU
            self.layers[i+1] = a
        return self.layers[-1]"""

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


def get_state(checkpoint_pos, player_pos, angle, speed, discretisations, thrust_relatif=False, prev_thrust=0):
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




def bot_dqn(player_send_q, player_receive_q, model_weigths = poids):

    model = Neural_Network_Git(5,9,[32,32], weights_biases=poids)

    t = 0
    a, b = 0, 0
    x, y = 0, 0

    discretisations_etat, discretisations_action = None, (3, 3)
    angle = None
    thrust_relatif = True

    prev_thrust = 100


    while True:
        try : 
            t += 1
            ax, ay = x, y
            try:
                x, y, next_checkpoint_x, next_checkpoint_y, next_checkpoint_dist, next_checkpoint_angle = [int(i) for i in player_receive_q.get().split()]
            except:
                player_receive_q.task_done()
                exit(0)
            player_receive_q.task_done()

            if angle is None:
                angle = angle_to((x, y), (next_checkpoint_x, next_checkpoint_y))
            else:
                if next_checkpoint_angle > 18:
                    next_checkpoint_angle = 18
                elif next_checkpoint_angle < -18:
                    next_checkpoint_angle = -18

                angle += next_checkpoint_angle

            state = get_state((next_checkpoint_x, next_checkpoint_y), (x, y), angle, (x - ax, y - ay), discretisations_etat, thrust_relatif=thrust_relatif, prev_thrust=prev_thrust)

             # Prédiction de l'action
            action_values = model.compute_output(state)
            action = np.argmax(action_values)

            target_x, target_y, thrust, prev_thrust = unpack_action(action, (x, y), angle, discretisations_action, thrust_relatif=True,prev_thrust = 100)
            player_send_q.put(f"{target_x} {target_y} {thrust if t != 1 else 'BOOST'}")
        
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            break

        
