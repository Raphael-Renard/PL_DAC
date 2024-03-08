import math
import pickle

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


def discretiser_angle(angle):
    # Limits to a 90° vision
    if angle < -45:
        angle = -45
    elif angle > 45:
        angle = 45

    # Discretize into 9 values ranging from -4 to 4
    angle //= 10
    if angle < 0:
        angle -= 1

    return angle


def discretiser_etat(checkpoint_pos, player_pos, angle, velocity):
    max_dist = 8000
    distance_intervals = np.exp(np.log(max_dist - 800) * np.arange(.1, 1.1, .1))

    distance_to_checkpoint = distance_to(checkpoint_pos, player_pos) - 800
    dcheckpoint_disc = 0
    while dcheckpoint_disc < 9 and distance_to_checkpoint < distance_intervals[dcheckpoint_disc]:
        dcheckpoint_disc += 1

    checkpoint_angle = angle_to(player_pos, checkpoint_pos)
    angle_to_checkpoint = diff_angle(angle, checkpoint_angle)

    acheckpoint_disc = discretiser_angle(angle_to_checkpoint)

    velocity_length = min(500, int(distance_to((0, 0), velocity)))
    velocity_length //= 100  # 500 values / 100 -> 5 values

    velocity_angle = angle_to(player_pos, player_pos + velocity)
    angle_to_velocity = diff_angle(angle, velocity_angle)
    avelocity_disc = discretiser_angle(angle_to_velocity)

    pol_next_checkpoint = (dcheckpoint_disc, acheckpoint_disc)  # 10*10 = 100 values
    pol_velocity = (velocity_length, avelocity_disc)  # 5*10 = 50 values

    return pol_next_checkpoint, pol_velocity  # 100 * 50 = 5000 values


def decode_action(action, angle, player_pos):
    angles = [-18, -11, -6, -3, -1, 1, 3, 6, 11, 18]

    thrust = (action // 10) * 25
    angle = (angle + angles[action % 10]) % 360

    target_x = player_pos[0] + 10000 * math.cos(math.radians(angle))
    target_y = player_pos[1] + 10000 * math.sin(math.radians(angle))

    return int(target_x), int(target_y), thrust


def bot_qlearning(player_send_q, player_receive_q):
    # Using queues to communicate with the main process instead of stdin/stdout

    t = 0
    x, y = 0, 0

    with open("../Training/q_table.pkl", "rb") as f:
        qtable = pickle.load(f)

    angle = None

    # game loop
    while True:
        try:
            t += 1

            ax, ay = x, y

            x, y, next_checkpoint_x, next_checkpoint_y, next_checkpoint_dist, next_checkpoint_angle = [int(i) for i in player_receive_q.get().split()]
            player_receive_q.task_done()

            if angle is None:
                angle = next_checkpoint_angle
            else:
                if next_checkpoint_angle > 18:
                    next_checkpoint_angle = 18
                elif next_checkpoint_angle < -18:
                    next_checkpoint_angle = -18

                angle += next_checkpoint_angle

            etat = discretiser_etat((next_checkpoint_x, next_checkpoint_y), (x, y), angle, (x - ax, y - ay))

            if etat in qtable:
                action = np.argmax(qtable[etat])
            else:
                action = np.random.randint(0, 50)

            target_x, target_y, thrust = decode_action(action, angle, (x, y))

            player_send_q.put(f"{target_x} {target_y} {thrust if t != 1 else 'BOOST'}")

        except Exception as e:
            print(f"Error: {e}")
            break
