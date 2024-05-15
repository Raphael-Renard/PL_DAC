import codecs
import math
import pickle
import gzip

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


