import random

import pygame
from player import Player


def get_possible_intervals(checkpoints, x):
    intervals = []
    buffer = 250
    ymin = 150
    for checkpoint in sorted(checkpoints, key=lambda c: c.y):
        if abs(checkpoint.x - x) < buffer:
            ymax = checkpoint.y - buffer
            if ymax > ymin:
                intervals.append((ymin, ymax))
            ymin = checkpoint.y + buffer
    if 750 > ymin:
        intervals.append((ymin, 750))
    return intervals


def generate_checkpoints():
    checkpoints = []
    for _ in range(random.randint(4, 6)):
        possible_intervals = []
        while not possible_intervals:
            x = random.randint(150, 1450)
            possible_intervals = get_possible_intervals(checkpoints, x)

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

    checkpoints = generate_checkpoints()

    players = []

    for i, (player_send_q, player_receive_q) in enumerate(player_queues):
        players.append(Player(i+1, player_names[i], checkpoints, player_send_q, player_receive_q, screen=screen, colour=player_colours[i], verbose=player_verbose))

    while running:
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
            end = player.update(dt)
            player.draw()
            if end:
                running = False

        # flip() the display to put your work on screen
        pygame.display.flip()

        # limits FPS to 60
        # dt is delta time in seconds since last frame, used for framerate-
        # independent physics.
        dt = clock.tick(30) / 1000

    end_game(True)


def game_loop_terminal(player_queues, player_names, player_verbose):
    checkpoints = generate_checkpoints()

    players = []

    for i, (player_send_q, player_receive_q) in enumerate(player_queues):
        players.append(Player(i+1, player_names[i], checkpoints, player_send_q, player_receive_q, verbose=player_verbose))

    win = []

    while not win:
        for player in players:
            end = player.update(1 / 60)
            if end:
                win.append(end)

    return win


def end_game(gui):
    if gui:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
    exit()
