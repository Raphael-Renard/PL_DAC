import codecs
import math
import pickle
import gzip
import sys

import numpy as np

encoded_table = """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"""

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

    disc_angle = 0

    while angle > angle_intervals[disc_angle]:
        disc_angle += 1

    # print(angle, disc_angle, angle_intervals)

    return disc_angle


def discretiser_distance(distance, nb_discretisations, min_distance=800, max_distance=10000, log=True):
    if log:
        distance_intervals = np.round(np.exp(np.log(max_distance - min_distance) * np.arange(0, 1.1, 1 / nb_discretisations))[1:])
    else:
        distance_intervals = np.linspace(min_distance, max_distance, nb_discretisations + 1)[1:]

    # print(distance, distance_intervals)

    disc_dist = 0
    while distance - min_distance > distance_intervals[disc_dist] and disc_dist < (nb_discretisations - 1):
        disc_dist += 1

    # print(distance, disc_dist, distance_intervals)

    return disc_dist


def discretiser_etat(checkpoint_pos, player_pos, angle, speed, discretisations=(9, 9, 9, 9)):
    dist_checkpoint = distance_to(player_pos, checkpoint_pos)
    disc_dist_checkpoint = discretiser_distance(dist_checkpoint, discretisations[0])

    checkpoint_angle = angle_to(player_pos, checkpoint_pos)
    angle_to_checkpoint = diff_angle(angle, checkpoint_angle)

    acheckpoint_disc = discretiser_angle(angle_to_checkpoint, discretisations[1])

    speed_length = distance_to((0, 0), speed)
    disc_speed_length = discretiser_distance(speed_length, discretisations[2], log=False, min_distance=0, max_distance=500)

    speed_angle = angle_to(player_pos, player_pos + speed)
    angle_to_speed = diff_angle(angle, speed_angle)
    aspeed_disc = discretiser_angle(angle_to_speed, discretisations[3])

    pol_next_checkpoint = (disc_dist_checkpoint, acheckpoint_disc)
    pol_speed = (disc_speed_length, aspeed_disc)

    return pol_next_checkpoint, pol_speed



def unpack_action(action: int, player_pos, angle, discretisations_action):
    nb_par_cote = discretisations_action[0] // 2
    side_intervals = np.round(np.exp(np.log(18) * np.arange(0, 1.1, 1 / nb_par_cote))[1:])
    angles = np.concatenate((-side_intervals[::-1], np.array([0]) if discretisations_action[0] % 2 == 1 else np.array(None), side_intervals))

    # dthrusts = np.round(np.linspace(-50, 50, discretisations_action[1]))

    # print(f"{self.nb_actions=}, {self.discretisations_action=}, {action=}")

    # thrust = self.prev_thrust + dthrusts[action // self.discretisations_action[0]]
    # thrust = max(0, min(100, thrust))

    # thrust = 100 if action // discretisations_action[0] != 0 else -50

    thrusts = np.round(np.linspace(0, 100, discretisations_action[1]))
    thrust = int(thrusts[action // discretisations_action[0]])

    # prev_thrust = thrust

    angle = (angle + angles[action % len(angles)]) % 360

    target_x = player_pos[0] + 10000 * math.cos(math.radians(angle))
    target_y = player_pos[1] + 10000 * math.sin(math.radians(angle))

    return round(target_x), round(target_y), thrust

qtable = pickle.loads(gzip.decompress(codecs.decode(encoded_table.encode(), "base64")))

t = 0
x, y = 0, 0

angle = None

# game loop
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

    etat = discretiser_etat((next_checkpoint_x, next_checkpoint_y), (x, y), angle, (x - ax, y - ay), discretisations=(9, 9, 9, 9))

    if etat in qtable:
        action = np.argmax(qtable[etat])
    else:
        action = 12
        print(etat, file=sys.stderr)

    print((x, y), (x - ax, y - ay), file=sys.stderr)
    target_x, target_y, thrust = unpack_action(action, (x, y), (angle), (5, 3))

    print(f"{target_x} {target_y} {thrust if t != 1 else 'BOOST'}")
