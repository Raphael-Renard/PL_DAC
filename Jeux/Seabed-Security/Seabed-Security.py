# Skeleton from SS_Starter.py from the game's official GitHub (see Game-implementation.md)
import math
import random
from typing import List, NamedTuple, Dict


# Define the data structures as namedtuples
class Vector(NamedTuple):
    x: int
    y: int


class FishDetail(NamedTuple):
    color: int
    type: int


class Fish(NamedTuple):
    fish_id: int
    pos: Vector
    speed: Vector
    detail: FishDetail


class RadarBlip(NamedTuple):
    fish_id: int
    dir: str


class Drone(NamedTuple):
    drone_id: int
    pos: Vector
    dead: bool
    battery: int
    scans: List[int]


def distance(e1, e2):
    return math.sqrt((e1.x - e2.x)**2 + (e1.y - e2.y)**2)


remontee: Dict[int, bool] = {}
target: Dict[int, Vector] = {}
light_cooldown: Dict[int, int] = {}
fish_details: Dict[int, FishDetail] = {}

fish_count = int(input())
for _ in range(fish_count):
    fish_id, color, _type = map(int, input().split())
    fish_details[fish_id] = FishDetail(color, _type)

t = 0
# game loop
while True:
    t += 1
    my_scans: List[int] = []
    foe_scans: List[int] = []
    drone_by_id: Dict[int, Drone] = {}
    my_drones: List[Drone] = []
    foe_drones: List[Drone] = []
    visible_fish: List[Fish] = []
    my_radar_blips: Dict[int, List[RadarBlip]] = {}

    my_score = int(input())
    foe_score = int(input())

    my_scan_count = int(input())
    for _ in range(my_scan_count):
        fish_id = int(input())
        my_scans.append(fish_id)

    foe_scan_count = int(input())
    for _ in range(foe_scan_count):
        fish_id = int(input())
        foe_scans.append(fish_id)

    my_drone_count = int(input())
    for _ in range(my_drone_count):
        drone_id, drone_x, drone_y, dead, battery = map(int, input().split())
        pos = Vector(drone_x, drone_y)
        drone = Drone(drone_id, pos, dead == '1', battery, [])
        drone_by_id[drone_id] = drone
        my_drones.append(drone)
        my_radar_blips[drone_id] = []

        if drone_id not in remontee:
            remontee[drone_id] = False
            target[drone_id] = Vector(drone_x, 10000)
            light_cooldown[drone_id] = 7

    foe_drone_count = int(input())
    for _ in range(foe_drone_count):
        drone_id, drone_x, drone_y, dead, battery = map(int, input().split())
        pos = Vector(drone_x, drone_y)
        drone = Drone(drone_id, pos, dead == '1', battery, [])
        drone_by_id[drone_id] = drone
        foe_drones.append(drone)

    drone_scan_count = int(input())
    for _ in range(drone_scan_count):
        drone_id, fish_id = map(int, input().split())
        drone_by_id[drone_id].scans.append(fish_id)

    visible_fish_count = int(input())
    for _ in range(visible_fish_count):
        fish_id, fish_x, fish_y, fish_vx, fish_vy = map(int, input().split())
        pos = Vector(fish_x, fish_y)
        speed = Vector(fish_vx, fish_vy)
        visible_fish.append(Fish(fish_id, pos, speed, fish_details[fish_id]))

    fish_dir_count = {"TL": 0, "BL": 0, "TR": 0, "BR": 0, "L": 0, "R": 0, "T": 0, "B": 0}
    my_radar_blip_count = int(input())
    for _ in range(my_radar_blip_count):
        drone_id, fish_id, dir = input().split()
        drone_id = int(drone_id)
        fish_id = int(fish_id)
        my_radar_blips[drone_id].append(RadarBlip(fish_id, dir))
        fish_dir_count[dir] += 1
        fish_dir_count[dir[0]] += 1
        fish_dir_count[dir[1]] += 1

    for drone in my_drones:
        d = 424
        wait = False

        light = 0
        light_cooldown[drone.drone_id] -= 1

        x = drone.pos.x
        y = drone.pos.y

        scannable_fish = False

        if 0 < x < 5000:
            intervalle = (800, 4601)
        else:
            intervalle = (5400, 9201)

        if remontee[drone.drone_id] and y < 500:
            remontee[drone.drone_id] = False
            while abs(x - target[drone.drone_id].x) <= 1500:
                target[drone.drone_id] = Vector(random.randint(intervalle[0], intervalle[1]), 1000)

        elif not remontee[drone.drone_id] and y > 9200:
            remontee[drone.drone_id] = True
            while abs(x - target[drone.drone_id].x) <= 1500:
                target[drone.drone_id] = Vector(random.randint(intervalle[0], intervalle[1]), 9000)

        if x == target[drone.drone_id].x and y == target[drone.drone_id].y:
            if y == 1000:
                target[drone.drone_id] = Vector(x, 10000)
            else:
                target[drone.drone_id] = Vector(x, 0)

        for fish in visible_fish:
            if fish.fish_id not in my_scans:
                scannable_fish = True

        # if x <= 500:
        #     if fish_dir_count["L"]:
        #         target_x = x - d
        #     else:
        #         target_x = x + d
        # else:
        #     if fish_dir_count["R"]:
        #         target_x = x + d
        #     else:
        #         target_x = x - d

        # if y <= 500:
        #     if fish_dir_count["T"]:
        #         target_y = y - d
        #     else:
        #         target_y = y + d
        # else:
        #     if fish_dir_count["B"]:
        #         target_y = y + d
        #     else:
        #         target_y = y - d

        if light_cooldown[drone.drone_id] == 0 or (light_cooldown[drone.drone_id] <= 3 and scannable_fish):
            light = 1
            light_cooldown[drone.drone_id] = 6

        target_x, target_y = target[drone.drone_id].x, target[drone.drone_id].y

        if wait:
            print(f"WAIT {light} {'Hi' if t < 5 else ''}{'GLHF' if 5 < t < 10 else ''}")
        else:
            print(f"MOVE {target_x} {target_y} {light} {'Hi' if t < 5 else ''}{'GHLF' if 5 < t < 10 else ''}")
