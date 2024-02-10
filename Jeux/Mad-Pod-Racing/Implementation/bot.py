def bot(player_send_q, player_receive_q):
    # Using queues to communicate with the main process instead of stdin/stdout

    t = 0
    x, y = 0, 0
    # game loop
    boost = True
    while True:
        try:
            t += 1
            # next_checkpoint_x: x position of the next check point
            # next_checkpoint_y: y position of the next check point
            # next_checkpoint_dist: distance to the next checkpoint
            # next_checkpoint_angle: angle between your pod orientation and the direction of the next checkpoint
            ax, ay = x, y

            x, y, next_checkpoint_x, next_checkpoint_y, next_checkpoint_dist, next_checkpoint_angle = [int(i) for i in player_receive_q.get().split()]
            player_receive_q.task_done()

            dx, dy = abs(ax - x), abs(ay - y)

            # opponent_x, opponent_y = [int(i) for i in input().split()]

            target_x = next_checkpoint_x
            target_y = next_checkpoint_y

            # Write an action using print
            # To debug: print("Debug messages...", file=sys.stderr, flush=True)

            # You have to output the target position
            # followed by the power (0 <= thrust <= 100)
            # i.e.: "x y thrust"
            thrust = 100

            if abs(next_checkpoint_angle) > 45:
                thrust -= abs(next_checkpoint_angle) - 20
            else:
                if next_checkpoint_dist < 3000:
                    thrust -= int((3000 - next_checkpoint_dist) / 30)

                # thrust = max(thrust, 50)

            thrust = max(thrust, 0)

            player_send_q.put(f"{target_x} {target_y} {thrust}")

        except Exception as e:
            print(f"Error: {e}")
            break
