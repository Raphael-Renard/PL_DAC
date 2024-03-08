import random

import pygame
from player import Player


def get_possible_intervals(checkpoints, x, gui):
    intervals = []
    if gui:
        buffer = 250
        ymin = 150
    else:
        buffer = 2500
        ymin = 1500
    for checkpoint in sorted(checkpoints, key=lambda c: c.y):
        if abs(checkpoint.x - x) < buffer:
            ymax = checkpoint.y - buffer
            if ymax > ymin:
                intervals.append((ymin, ymax))
            ymin = checkpoint.y + buffer
    if gui:
        if 750 > ymin:
            intervals.append((ymin, 750))
    else:
        if 7500 > ymin:
            intervals.append((ymin, 7500))
    return intervals


def generate_checkpoints(gui=False):
    checkpoints = []
    for _ in range(random.randint(4, 6)):
        possible_intervals = []
        while not possible_intervals:
            if gui:
                x = random.randint(150, 1450)
            else:
                x = random.randint(1500, 14500)
            possible_intervals = get_possible_intervals(checkpoints, x, gui)

        chosen_interval = random.choice(possible_intervals)
        y = random.randint(chosen_interval[0], chosen_interval[1])

        checkpoints.append(pygame.Vector2(x, y))

    return checkpoints


def game_loop_gui(player_queues, player_names, player_colours, player_verbose):
    # pygame setup
    pygame.init()

    screen = pygame.display.set_mode((1600, 900))
    clock = pygame.time.Clock()
    running = True
    dt = 0

    checkpoints = generate_checkpoints(True)

    players = []

    for i, (player_send_q, player_receive_q) in enumerate(player_queues):
        players.append(Player(i + 1, player_names[i], checkpoints, player_send_q, player_receive_q, screen=screen, colour=player_colours[i], verbose=player_verbose))

    t = 0
    while running:
        t += 1
        pygame.font.init()
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # fill the screen with a color to wipe away anything from last frame
        screen.fill("#EE913D")

        # draw the checkpoints
        for i, checkpoint in enumerate(checkpoints):
            pygame.draw.circle(screen, "#D14A1E", checkpoint, 60)
            pygame.draw.circle(screen, "#DAA295", checkpoint, width=5, radius=60)
            myfont = pygame.font.Font(None, 36)
            label = myfont.render(str(i), 1, "white")
            screen.blit(label, checkpoint - pygame.Vector2(9, 9))

        for player in players:
            end = player.update(dt, True)
            player.draw()
            if end:
                running = False

        # Display player colour and name at top left corner
        for i, player in enumerate(players):
            myfont = pygame.font.Font(None, 36)
            label = myfont.render(f"{player.name}", 1, player.colour)
            screen.blit(label, (10, 10 + i * 40))

        # flip() the display to put your work on screen
        pygame.display.flip()

        # limits FPS to 60
        # dt is delta time in seconds since last frame, used for framerate-
        # independent physics.
        clock.tick(10)
        dt = 1/10

    print(f"{t=}")
    end_game(True)


def game_loop_terminal(player_queues, player_names, player_verbose):
    checkpoints = generate_checkpoints()

    players = []

    for i, (player_send_q, player_receive_q) in enumerate(player_queues):
        players.append(Player(i + 1, player_names[i], checkpoints, player_send_q, player_receive_q, verbose=player_verbose))

    win = []
    t = 0
    while not win:
        t += 1
        for player in players:
            end = player.update(1)
            if end:
                win.append(end)

        if t == 1000:
            t = "timeout"
            break

    # print(f"{t=}")
    return win


def end_game(gui):
    if gui:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
    exit()
