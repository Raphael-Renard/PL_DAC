import random

class Aleatoire():
    def __init__(self, state):
        self.state = state
    def give_move(self):
        return random.choice(list(self.state.get_possible_moves()))