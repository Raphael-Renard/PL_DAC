import numpy as np
import pygame

class Game:
    def __init__(self, player_send_q, player_receive_q, checkpoints):
        self.player_send_q = player_send_q
        self.player_receive_q = player_receive_q
        self.nb_actions = 3  # tourner à gauche, tourner à droite, puissance
        self.nb_states = 100  # il faudrait discrétiser la map ?
        self.checkpoints = checkpoints
        self.current_state = (0, 0, checkpoints[0][0], checkpoints[0][1], 0)  # Initialisation de l'état
        self.num_checkpoints = len(checkpoints)
        self.current_checkpoint = 0  # Correction de 'current_checkpoints' à 'current_checkpoint'

    def reset(self):
        # Réinitialiser l'état du jeu et retourner l'état initial
        initial_x = 0
        initial_y = 0
        initial_next_checkpoint_x = self.checkpoints[0][0]
        initial_next_checkpoint_y = self.checkpoints[0][1]
        initial_angle = 0  

        self.current_state = (initial_x, initial_y, initial_next_checkpoint_x, initial_next_checkpoint_y, initial_angle)
        return self.current_state  # Assurez-vous de retourner l'état initial après la réinitialisation

    def step(self, action):
        self.update_player_position(action)
        # Si un joueur atteint un checkpoint

        if self.checkpoint_reached():
            self.current_checkpoint += 1

            # On vérifie si tous les checkpoints ont été atteints
            if self.current_checkpoint == self.num_checkpoints:
                return self.get_state(), 1.0, True  # Récompense maximale et le jeu est terminé

        # On calule la récompense en fonction de la norme euclidienne entre le checkpoint et le state (plus près du prochain checkpoint est mieux)
        reward = 1.0 / (1.0 + np.linalg.norm(self.get_state_position - self.get_checkpoint()))

        return self.get_state_position(), reward, False
    
    def update_player_position(self, thrust):
        # fonction qui met a jour dans l'objet Game

    def checkpoint_reached(self):
        # Vérifiez si le joueur est suffisamment proche du prochain checkpoint
        return np.linalg.norm(self.get_state_position() - self.get_checkpoint()) < 20.0
    
    def get_state_position(self):
        return self.current_state[:2]

    def get_checkpoint(self):
        checkpoint = self.checkpoints[self.current_checkpoint]
        return checkpoint[0],checkpoint[1]


def q_learning(game, num_episodes=1000, alpha=0.1, gamma=0.99, epsilon=0.1):
    q_table = np.zeros((game.nb_states, game.nb_actions))
    for episode in range(num_episodes):
        state = game.reset()
        done = False

        while not done:
            if np.random.rand() < epsilon:
                action = np.random.randint(game.nb_actions)
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done = game.step(action)
            q_table[state, action] += alpha * (reward + gamma * np.max(q_table[next_state]) - q_table[state, action])
            state = next_state

    return q_table


def bot_q_learning(player_send_q, player_receive_q):
    game = Game()
    q_table = q_learning(game)

    t = 0
    x, y = 0, 0

    while True:
        try:
            t += 1
            ax, ay = x, y

            x, y, next_checkpoint_x, next_checkpoint_y, next_checkpoint_dist, next_checkpoint_angle = [int(i) for i in player_receive_q.get().split()]
            player_receive_q.task_done()

            current_state = x, y, next_checkpoint_x, next_checkpoint_y, next_checkpoint_dist, next_checkpoint_angle

            # Choix de l'action basée sur la table Q
            action = np.argmax(q_table[current_state])

            # Exécution de l'action
            # Vous devrez ajuster cela selon votre logique spécifique de déplacement du pod
            thrust = action
            game.step(thrust)

            # Envoie de la nouvelle position au jeu
            player_send_q.put(f"{game.get_state()[0]} {game.get_state()[1]} {thrust}")

        except Exception as e:
            print(f"Error: {e}")
            break