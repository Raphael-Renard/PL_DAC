import math
import traceback

import numpy as np
import torch
import torch.nn as nn


class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 2 * action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=-1)


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




def unpack_action(action, player_pos, angle, thrust_relatif=False, prev_thrust=0):
    dangle, thrust = action

    angle = (angle + dangle) % 360

    target_x = player_pos[0] + 10000 * math.cos(math.radians(angle))
    target_y = player_pos[1] + 10000 * math.sin(math.radians(angle))

    return round(target_x), round(target_y), int(thrust)


def bot_policy_gradient(player_send_q, player_receive_q, model_path="../Resultats/Policy_gradient_policy_network.pth"):
    # Using queues to communicate with the main process instead of stdin/stdout

    t = 0
    a, b = 0, 0
    x, y = 0, 0

    # Load policy network
    model = PolicyNetwork(4, 2)
    model.load_state_dict(torch.load(model_path))

    angle = None

    discretisations_etat, discretisations_action = None, (3, 3)
    thrust_relatif = False

    prev_thrust = 100

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

            etat = get_state((next_checkpoint_x, next_checkpoint_y), (x, y), angle, (x - ax, y - ay), discretisations_etat, thrust_relatif=thrust_relatif, prev_thrust=prev_thrust)

            b += 1

            state_tensor = torch.FloatTensor(etat)

            action_parameters = model(state_tensor).reshape((4,))

            # action_parameters = action_parameters[0]

            action_mean = action_parameters[::2]
            action_std = action_parameters[1::2]

            # print(f"action_mean: {action_mean}, action_std: {action_std}")

            action_std = torch.clamp(action_std, 1e-6, 1)
            action_dist = torch.distributions.Normal(action_mean, action_std)

            # print(action_dist)

            action = action_dist.sample()
            print(f"action: {action}")


            target_x, target_y, thrust = unpack_action(action, (x, y), angle)

            player_send_q.put(f"{target_x} {target_y} {thrust if t != 1 else 'BOOST'}")

        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            break
