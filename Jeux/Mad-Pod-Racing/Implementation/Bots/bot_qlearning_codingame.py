import codecs
import math
import pickle
import gzip

import numpy as np

encoded_table = """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\n'
"""


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


qtable = pickle.loads(gzip.decompress(codecs.decode(encoded_table.encode(), "base64")))

t = 0
a, b = 0, 0
x, y = 0, 0

angle = None

discretisations_etat, discretisations_action = (5, 5, 5, 5, 5), (5, 5)
thrust_relatif = True

prev_thrust = 100

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

    etat = discretiser_etat((next_checkpoint_x, next_checkpoint_y), (x, y), angle, (x - ax, y - ay), discretisations=discretisations_etat, thrust_relatif=thrust_relatif, prev_thrust=prev_thrust)

    if etat in qtable:
        action = np.argmax(qtable[etat])
    else:
        action = discretisations_action[0] * discretisations_action[1] - discretisations_action[0] // 2

    target_x, target_y, thrust, prev_thrust = unpack_action(action, (x, y), angle, discretisations_action, thrust_relatif=thrust_relatif, prev_thrust=prev_thrust)

    print(f"{target_x} {target_y} {thrust if t != 1 else 'BOOST'}")


