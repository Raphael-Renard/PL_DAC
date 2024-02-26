def bot_simple(player_send_q, player_receive_q):
    # Using queues to communicate with the main process instead of stdin/stdout

    t = 0
    x, y = 0, 0

    # game loop
    while True:
        try:
            t += 1
            ax, ay = x, y

            x, y, next_checkpoint_x, next_checkpoint_y, next_checkpoint_dist, next_checkpoint_angle = [int(i) for i in player_receive_q.get().split()]
            player_receive_q.task_done()

            dx, dy = abs(ax - x), abs(ay - y)

            target_x = next_checkpoint_x
            target_y = next_checkpoint_y

            thrust = 100

            if abs(next_checkpoint_angle) > 90:
                thrust = 0

            player_send_q.put(f"{target_x} {target_y} {thrust}")

        except Exception as e:
            print(f"Error: {e}")
            break
