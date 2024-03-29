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


def discretiser_distance(distance, nb_discretisations, max_distance=10000, log=True):
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



def unpack_action(action: int, player_pos, angle, discretisations_action):
    print(action, action // discretisations_action[0])
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


def bot_qlearning(player_send_q, player_receive_q, qtable_path="q_table"):
    # Using queues to communicate with the main process instead of stdin/stdout

    t = 0
    x, y = 0, 0

    with open(f"../Training/{qtable_path}.pkl", "rb") as f:
        qtable = pickle.load(f)

    angle = None

    # game loop
    while True:
        try:
            t += 1

            ax, ay = x, y

            try:
                x, y, next_checkpoint_x, next_checkpoint_y, next_checkpoint_dist, next_checkpoint_angle = [int(i) for i in player_receive_q.get().split()]
            except:
                player_receive_q.task_done()
                exit(0)
            player_receive_q.task_done()

            if angle is None:
                angle = next_checkpoint_angle
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
                action = np.random.randint(0, 15)

            target_x, target_y, thrust = unpack_action(action, (x, y), (angle), (5, 3))

            player_send_q.put(f"{target_x} {target_y} {thrust if t != 1 else 'BOOST'}")

        except Exception as e:
            print(f"Error: {e}")
            break
