from io import StringIO
from threading import Event, Thread
from queue import Queue
# import pygame

from a_renommer import *


def game_loop():
    # pygame setup
    pygame.init()

    screen = pygame.display.set_mode((1600, 900))
    clock = pygame.time.Clock()
    running = True
    dt = 0

    checkpoints = generate_checkpoints()

    player = Player(screen, pygame.Vector2(checkpoints[0]), checkpoints, player_send_q, player_receive_q)

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

        player.update(dt)
        player.draw()

        # flip() the display to put your work on screen
        pygame.display.flip()



        # limits FPS to 60
        # dt is delta time in seconds since last frame, used for framerate-
        # independent physics.
        dt = clock.tick(60) / 1000

    pygame.quit()


if __name__ == "__main__":
    oldstdin = sys.stdin
    sys.stdin = StringIO()

    threads = []

    player_send_q = Queue()
    player_receive_q = Queue()
    # launch player thread on ../Mad-Pod-Racing.py in background
    threads.append(Thread(target=player_loop, args=(player_send_q, player_receive_q)))
    threads[-1].start()
    print("Player thread started")
    game_loop()
