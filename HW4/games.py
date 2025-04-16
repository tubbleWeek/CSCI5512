import numpy as np
from itertools import combinations

class PrisonersDilema:
    def __init__(
        self
    ):
        self.moves = ["Testify", "Refuse"]
        self.value_dict = {
            ("Refuse", "Refuse"): (3.0, 3.0),
            ("Testify", "Refuse"): (5.0, 0.0),
            ("Refuse", "Testify"): (0.0, 5.0),
            ("Testify", "Testify"): (1.0, 1.0)
                           }
        

        # self.time = 0
        self.move_list = []

        def playout(agentA, agentB):
            agentA_action = agentA.get_action(self)
            self.move_list.append(agentA_action)
            # time += 1
            agentB_action = agentB.get_action(self)
            self.move_list.append(agentB_action)
            return self.value_dict[(agentA_action, agentB_action)]
        
        def get_moves():
            return self.move_list
        
        def get_values():
            return self.value_dict
        
class Chicken:
    def __init__(
        self
    ):
        self.moves = ["Swerve", "Straight"]
        self.value_dict = {
            ("Swerve", "Swerve"): (3.0, 3.0),
            ("Swerve", "Straight"): (1.5, 3.5),
            ("Straight", "Swerve"): (3.5, 1.5),
            ("Straight", "Straight"): (1.0, 1.0)
                           }
        

        # self.time = 0
        self.move_list = []

        def playout(agentA, agentB):
            agentA_action = agentA.get_action(self)
            self.move_list.append(agentA_action)
            # time += 1
            agentB_action = agentB.get_action(self)
            self.move_list.append(agentB_action)
            return self.value_dict[(agentA_action, agentB_action)]
        
        def get_moves():
            return self.move_list
        
        def get_values():
            return self.value_dict

class MovieSelection:
    def __init__(
        self
    ):
        self.moves = ["Action", "Comedy"]
        self.value_dict = {
            ("Action", "Action"): (3.0, 2.0),
            ("Action", "Comedy"): (0.0, 0.0),
            ("Comedy", "Action"): (0.0, 0.0),
            ("Comedy", "Comedy"): (2.0, 3.0)
                           }
        

        # self.time = 0
        self.move_list = []

        def playout(agentA, agentB):
            agentA_action = agentA.get_action(self)
            self.move_list.append(agentA_action)
            # time += 1
            agentB_action = agentB.get_action(self)
            self.move_list.append(agentB_action)
            return self.value_dict[(agentA_action, agentB_action)]
        
        def get_moves():
            return self.moves
        
        def get_plays():
            return self.move_list
        
        def get_values():
            return self.value_dict

class TitForTat:
    def __init__(
        self
    ):
          # self.time = 0
        self.opp_movelist = []

    def get_action(self, game):
        move_sequence = game.get_plays()
        action = None
        if len(self.opp_movelist) == 0 and len(move_sequence) == 0:
            action = np.random.choice(game.get_moves())
        else:
            self.opp_movelist.append(move_sequence[len(move_sequence)])
            action = self.opp_movelist[len(self.opp_movelist)]
        assert type(action) == str
        return action
    
    def clear_movelist(self):
        self.opp_movelist = []
class Bully:
    def __init__(
        self
    ):
        # self.time = 0
        self.score_dict = {}

    def get_action(self, game):
        # move_sequence = game.get_plays()
        possible_moves = []
        for move in combinations(game.get_move()):
            possible_moves.append(move)
        for move in possible_moves:
            reward = game.playout(move)
        
        action = None
        if len(self.opp_movelist) == 0 and len(move_sequence) == 0:
            action = np.random.choice(game.get_moves())
        else:
            self.opp_movelist.append(move_sequence[len(move_sequence)])
            action = self.opp_movelist[len(self.opp_movelist)]
        assert type(action) == str
        return action
    
    def clear_movelist(self):
        self.opp_movelist = []

class GodFather:
    def __init__(
        self
    ):
          # self.time = 0
        self.opp_movelist = []

    def get_action(self, game):
        move_sequence = game.get_plays()
        action = None
        if len(self.opp_movelist) == 0 and len(move_sequence) == 0:
            action = np.random.choice(game.get_moves())
        else:
            self.opp_movelist.append(move_sequence[len(move_sequence)])
            action = self.opp_movelist[len(self.opp_movelist)]
        assert type(action) == str
        return action
    
    def clear_movelist(self):
        self.opp_movelist = []
class FictitiousPlay:
    def __init__(
        self
    ):
          # self.time = 0
        self.opp_movelist = []

    def get_best_response_to_play_count(A, play_count):
        utilities = A @ play_count
        return np.random.choice(np.argwhere(utilities == np.max(utilities)).transpose()[0])
    
    def update_play_count(play_count, play):
        extra_play = np.zeros(play_count.shape)
        extra_play[play] = 1
        return play_count + extra_play
    
    def fictitious_play(self, A, B, iterations, play_counts = None):

        if play_counts is None:
            play_counts = [np.array([0 for _ in range(dimension)]) for dimension in A.shape]

        yield play_counts

        for repetition in range(iterations):
            plays = [
                self.get_best_response_to_play_count(matrix, play_count)
                for matrix, play_count in zip((A, B.transpose()), play_counts[::-1])
            ]

            play_counts = [
                self.update_play_count(play_count, play)
                for play_count, play in zip(play_counts, plays)
            ]
            yield play_counts
            
    def get_action(self, game):
        move_sequence = game.get_plays()
        action = None
        if len(self.opp_movelist) == 0 and len(move_sequence) == 0:
            action = np.random.choice(game.get_moves())
        else:
            self.opp_movelist.append(move_sequence[len(move_sequence)])
            action = self.opp_movelist[len(self.opp_movelist)]
        assert type(action) == str
        return action

    def clear_movelist(self):
        self.opp_movelist = []


def main():
    # implement main game loop
    return None

if __name__ == "__main__":
    main()