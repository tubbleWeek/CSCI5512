import numpy as np
from itertools import combinations

class PrisonersDilema:
    def __init__(self):
        self.moves = ["Testify", "Refuse"]
        self.value_dict = {
            ("Refuse", "Refuse"): (3.0, 3.0),
            ("Testify", "Refuse"): (5.0, 0.0),
            ("Refuse", "Testify"): (0.0, 5.0),
            ("Testify", "Testify"): (1.0, 1.0)
        }
        self.move_list = []

    def playout(self, agent_a, agent_b):
        a_action = agent_a.get_action(self)
        b_action = agent_b.get_action(self)
        self.move_list.append(a_action)
        self.move_list.append(b_action)
        
        # Update agent knowledge after the round
        if hasattr(agent_a, 'update_history'):
            agent_a.update_history(self, b_action)
        if hasattr(agent_b, 'update_history'):
            agent_b.update_history(self, a_action)
        if hasattr(agent_a, 'update_beliefs'):
            agent_a.update_beliefs(self, b_action)
        if hasattr(agent_b, 'update_beliefs'):
            agent_b.update_beliefs(self, a_action)
            
        return self.value_dict[(a_action, b_action)]
    
    def get_moves(self):
        return self.moves
    
    def get_plays(self):
        return self.move_list
    
    def get_values(self):
        return self.value_dict
        
class Chicken:
    def __init__(self):
        self.moves = ["Swerve", "Straight"]
        self.value_dict = {
            ("Swerve", "Swerve"): (3.0, 3.0),
            ("Swerve", "Straight"): (1.5, 3.5),
            ("Straight", "Swerve"): (3.5, 1.5),
            ("Straight", "Straight"): (1.0, 1.0)
        }
        self.move_list = []

    def playout(self, agent_a, agent_b):
        a_action = agent_a.get_action(self)
        b_action = agent_b.get_action(self)
        self.move_list.append(a_action)
        self.move_list.append(b_action)
        
        # Update agent knowledge after the round
        if hasattr(agent_a, 'update_history'):
            agent_a.update_history(self, b_action)
        if hasattr(agent_b, 'update_history'):
            agent_b.update_history(self, a_action)
        if hasattr(agent_a, 'update_beliefs'):
            agent_a.update_beliefs(self, b_action)
        if hasattr(agent_b, 'update_beliefs'):
            agent_b.update_beliefs(self, a_action)
            
        return self.value_dict[(a_action, b_action)]
    
    def get_moves(self):
        return self.moves
    
    def get_plays(self):
        return self.move_list
    
    def get_values(self):
        return self.value_dict

class MovieSelection:
    def __init__(self):
        self.moves = ["Action", "Comedy"]
        self.value_dict = {
            ("Action", "Action"): (3.0, 2.0),
            ("Action", "Comedy"): (0.0, 0.0),
            ("Comedy", "Action"): (0.0, 0.0),
            ("Comedy", "Comedy"): (2.0, 3.0)
        }
        self.move_list = []

    def playout(self, agent_a, agent_b):
        a_action = agent_a.get_action(self)
        b_action = agent_b.get_action(self)
        self.move_list.append(a_action)
        self.move_list.append(b_action)
        
        # Update agent knowledge after the round
        if hasattr(agent_a, 'update_history'):
            agent_a.update_history(self, b_action)
        if hasattr(agent_b, 'update_history'):
            agent_b.update_history(self, a_action)
        if hasattr(agent_a, 'update_beliefs'):
            agent_a.update_beliefs(self, b_action)
        if hasattr(agent_b, 'update_beliefs'):
            agent_b.update_beliefs(self, a_action)
            
        return self.value_dict[(a_action, b_action)]
    
    def get_moves(self):
        return self.moves
    
    def get_plays(self):
        return self.move_list
    
    def get_values(self):
        return self.value_dict

class TitForTat:
    def __init__(self):
        self.opponent_moves = []
    
    def get_action(self, game):
        if not self.opponent_moves:  # First move
            return np.random.choice(game.get_moves())
        return self.opponent_moves[-1]
    
    def update(self, opponent_move):
        self.opponent_moves.append(opponent_move)
    
    def reset(self):
        self.opponent_moves = []

class Bully:
    def get_action(self, game):
        # Check the game type and return the aggressive move
        if isinstance(game, PrisonersDilema):
            return "Testify"
        elif isinstance(game, Chicken):
            return "Straight"
        elif isinstance(game, MovieSelection):
            return "Action"
        else:
            # Default to the first available move if unknown game
            return np.random.choice(game.get_moves())

class GodFather:
    def __init__(self):
        self.opponent_history = {}
        self.grace_periods = {
            PrisonersDilema: 2,
            Chicken: 1,
            MovieSelection: 3
        }

    def get_action(self, game):
        game_id = id(game)
        if game_id not in self.opponent_history:
            self.opponent_history[game_id] = {
                'opponent_moves': [],
                'grace_remaining': self.grace_periods.get(type(game), 1)
            }

        history = self.opponent_history[game_id]
        opponent_moves = history['opponent_moves']
        
        # Get opponent's last move if available
        last_opponent_move = opponent_moves[-1] if opponent_moves else None
        
        # Game-specific strategies
        if isinstance(game, PrisonersDilema):
            return self._prisoners_dilemma_strategy(last_opponent_move, history)
        elif isinstance(game, Chicken):
            return self._chicken_strategy(last_opponent_move, history)
        elif isinstance(game, MovieSelection):
            return self._movie_strategy(last_opponent_move, history)
        else:
            return game.moves[0]

    def _prisoners_dilemma_strategy(self, last_opponent_move, history):
        # Initial aggressive move
        if not last_opponent_move:
            return "Testify"
        
        # Forgive occasional defections within grace period
        if last_opponent_move == "Refuse":
            history['grace_remaining'] = self.grace_periods[PrisonersDilema]
            return "Refuse"
            
        # Retaliate against defections but allow recovery
        if history['grace_remaining'] > 0:
            history['grace_remaining'] -= 1
            return "Refuse"
            
        return "Testify"

    def _chicken_strategy(self, last_opponent_move, history):
        # Initial aggressive posture
        if not last_opponent_move:
            return "Straight"
        
        # Crush weakness but avoid mutual destruction
        if last_opponent_move == "Swerve":
            return "Straight"
            
        # Strategic retreat after opponent defiance
        history['grace_remaining'] = self.grace_periods[Chicken]
        return "Swerve"

    def _movie_strategy(self, last_opponent_move, history):
        # Establish dominance first
        if not last_opponent_move:
            return "Action"
        
        # Punish disobedience but allow reconciliation
        if last_opponent_move == "Comedy":
            if history['grace_remaining'] > 0:
                history['grace_remaining'] -= 1
                return "Comedy"
            return "Action"
            
        # Reward compliance
        history['grace_remaining'] = self.grace_periods[MovieSelection]
        return "Action"

    def update_history(self, game, opponent_move):
        game_id = id(game)
        if game_id in self.opponent_history:
            self.opponent_history[game_id]['opponent_moves'].append(opponent_move)
    
    def reset(self):
        self.opponent_history = {}

class FictitiousPlay:
    def __init__(self, player_index=0, smoothing=1e-5):
        self.history = {}  # {game_id: {opponent_moves: counts}}
        self.player_index = player_index  # 0 for row player, 1 for column
        self.smoothing = smoothing  # Dirichlet prior for probability estimation

    def get_action(self, game):
        game_id = id(game)
        opponent_moves = game.get_moves()
        
        # Initialize belief state for new games
        if game_id not in self.history:
            self.history[game_id] = {move: self.smoothing for move in opponent_moves}
        
        # Calculate empirical distribution with smoothing
        counts = self.history[game_id]
        total = sum(counts.values())
        beliefs = {move: count/total for move, count in counts.items()}
        
        # Calculate expected utilities for all possible actions
        utilities = {
            action: sum(
                prob * game.value_dict[(action if self.player_index == 0 else opp_action, 
                                      opp_action if self.player_index == 0 else action)][self.player_index]
                for opp_action, prob in beliefs.items()
            )
            for action in game.moves
        }
        
        # Return action with maximum expected utility
        return max(utilities.items(), key=lambda x: x[1])[0]

    def update_beliefs(self, game, opponent_move):
        game_id = id(game)
        if game_id not in self.history:
            self.history[game_id] = {move: self.smoothing for move in game.moves}
        self.history[game_id][opponent_move] += 1

    def reset(self):
        self.history = {}
    
def run_tournament(agents, games, num_rounds=100):
    results = {}
    for game_class in games:
        game_name = game_class.__name__
        results[game_name] = {}
        
        # Initialize score matrix for this game
        score_matrix = {(a, b): [0.0, 0.0] for a in agents for b in agents}
        
        for agent_a in agents:
            for agent_b in agents:
                # Reset agents for this matchup
                if hasattr(agent_a, 'reset'):
                    agent_a.reset()
                if hasattr(agent_b, 'reset'):
                    agent_b.reset()
                
                # Single game instance for 100 rounds
                game = game_class()
                total_a, total_b = 0.0, 0.0
                
                for _ in range(num_rounds):
                    payoff_a, payoff_b = game.playout(agent_a, agent_b)
                    total_a += payoff_a
                    total_b += payoff_b
                
                # Store averages for this game
                score_matrix[(agent_a, agent_b)] = [
                    total_a / num_rounds,
                    total_b / num_rounds
                ]
        
        results[game_name] = score_matrix
    
    return results

def print_results(results):
    for game_name, score_matrix in results.items():
        print(f"\nGame: {game_name}")
        agents = list({a for a, _ in score_matrix.keys()})
        
        # Print matrix header
        print("|            | " + " | ".join(f"{agent.__class__.__name__:^15}" for agent in agents) + " |")
        print("|------------|" + "|".join(["-----------------"] * len(agents)) + "|")
        
        # Print rows
        for agent_a in agents:
            row = [f"{agent_a.__class__.__name__:<11} |"]
            for agent_b in agents:
                alpha, beta = score_matrix[(agent_a, agent_b)]
                row.append(f" {alpha:.2f}, {beta:.2f} ")
            print("|" + "|".join(row) + "|")

# Updated main section
if __name__ == "__main__":
    agents = [
        TitForTat(),
        FictitiousPlay(),
        Bully(),
        GodFather()
    ]
    
    games = [PrisonersDilema, Chicken, MovieSelection]
    score_matrix = run_tournament(agents, games, num_rounds=1000)
    
    print("\nTournament Results Matrix (α, β):")
    print_results(score_matrix)