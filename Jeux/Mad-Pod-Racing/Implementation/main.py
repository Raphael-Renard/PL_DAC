from io import StringIO
from threading import Event, Thread
from queue import Queue
import sys

from player import *
from bot import *


def generate_checkpoints():
    checkpoints = []
    for _ in range(random.randint(4, 6)):
        checkpoint = pygame.Vector2(random.randint(150, 1450), random.randint(150, 750))
        while any(checkpoint.distance_to(checkpoint2) < 500 for checkpoint2 in checkpoints):
            checkpoint = pygame.Vector2(random.randint(100, 1500), random.randint(100, 800))
        checkpoints.append(checkpoint)

    return checkpoints


def game_loop():
    # pygame setup
    pygame.init()

    screen = pygame.display.set_mode((1600, 900))
    clock = pygame.time.Clock()
    running = True
    dt = 0

    checkpoints = generate_checkpoints()

    player = Player("Bot un peu nul de Luc", screen, checkpoints, player_send_q, player_receive_q)

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
        dt = clock.tick(30) / 1000

    end_game()


def end_game():
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    oldstdin = sys.stdin
    sys.stdin = StringIO()

    threads = []

    player_send_q = Queue()
    player_receive_q = Queue()

    threads.append(Thread(target=bot, args=(player_send_q, player_receive_q)))
    threads[-1].start()

    print("Player thread started")
    game_loop()
