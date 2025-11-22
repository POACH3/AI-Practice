"""
menace.py

Implements Donald Michie's MENACE (Machine Educable Noughts And Crosses Engine).

A reinforcement learning algorithm for small, discrete state spaces.
The entire statespace must be calculated in advance. The possible moves at
each state are associated with a probability distribution that is updated over
repeated iterations based on outcomes.

NOTES:
    fix save_model() so JSON is more readable.
    consider passing the player_position every move instead of in a constructor (combine models?)
"""
import os
import json
from matchbox import Matchbox


class Menace:
    """
    Represents a MENACE agent that learns optimal moves through reinforcement.
    """

    def __init__(self, **kwargs):
        """
        Constructor.

        Args:
            **kwargs (dict): Expected keys:
                - player_position (int): The play order position (player number) of the MENACE agent.
                - game_name (str): The name of the game being played.
                - states_and_moves (dict, optional): Maps game states to their legal moves. Optional if model already exists.
        """
        self.player_position = kwargs.get('player_position')
        self.game_name = kwargs['game_name']
        self.states_and_moves = kwargs.get('states_and_moves') # optional

        self.model_path = None                                 # optional alternative to default model
        self.matchboxes = {}

        if self.model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))  # this directory
            self.model_path = os.path.join(base_dir, f'models/{self.game_name}_player{self.player_position}_menace_model.json') # load default model for that game

        if os.path.exists(self.model_path):
            self.import_model(self.model_path)
            #FIXME - if there is an error in importing, create a new model
        elif self.states_and_moves is not None:
            self.create_model(self.player_position, self.game_name, self.states_and_moves)
            self.import_model(self.model_path)
        else:
            raise FileNotFoundError('A model for this game was not found.')


    def create_model(self, player_position, game_name, states_and_moves, initial_beads=3):
        """
        Sets up matchboxes based on given game states and moves, then writes a JSON file.

        Args:
            player_position (int): The play order position (player number) of the MENACE agent.
            game_name (string): The name of the game being played.
            states_and_moves (dict): Maps game states to their legal moves.
            initial_beads (int): The number of initial beads to use for each move.
        """
        for state, moves in states_and_moves.items():

            moves_and_beads = {}
            for move in moves:
                moves_and_beads[move] = initial_beads

            matchbox = Matchbox(str(state), moves_and_beads)
            self.matchboxes[state] = matchbox

        base_dir = os.path.dirname(os.path.abspath(__file__))  # this directory
        self.model_path = os.path.join(base_dir, f'models/{game_name}_player{player_position}_menace_model.json')
        self.save_model()


    def import_model(self, model_file):
        """
        Reads in a JSON file that represents the states and beads and initializes
        all the matchboxes for the game.

        Args:
            model_file (string): Path to the JSON file.
        """
        with open(model_file, 'r') as f:
            model_json = json.load(f)

        for state_str, moves_list in model_json.items():
            moves_and_beads = {}
            for item in moves_list:
                from_row, from_col = item["move"][0]
                to_row, to_col = item["move"][1]
                move_tuple = ((from_row, from_col), (to_row, to_col))
                moves_and_beads[move_tuple] = item["beads"]

            self.matchboxes[state_str] = Matchbox(state_str, moves_and_beads)


    def save_model(self):
        """
        Saves the model as JSON, writing over the original.
        """
        model_json = {}
        for state, matchbox in self.matchboxes.items():
            moves_list = []
            for move, beads in matchbox.moves_and_beads.items():
                (from_row, from_col), (to_row, to_col) = move
                moves_list.append({
                    "move": [[from_row, from_col], [to_row, to_col]],
                    "beads": beads
                })
            model_json[state] = moves_list

        with open(self.model_path, 'w') as f:
            json.dump(model_json, f, indent=4)


    def get_move(self, board_state):
        """
        Gets the move to make, given the board state.

        Args:
            board_state (str): A string representing the board state.

        Returns:
            move (tuple): The move to make.
        """
        matchbox = self.matchboxes.get(board_state)

        if matchbox is None:
            raise Exception(f'No matchbox found for board state:\n{board_state}')

        move = matchbox.draw_bead()
        return move


    def game_report(self, game_history, player_position, winner_position):
        """
        End of game report. Necessary for model training.

        Args:
            game_history (list): A list of (state, move) tuples representing the game history.
            player_position (int): The play order position (player number) of the MENACE agent.
            winner_position (int): The play order position (player number) of the winner of the game.
        """
        self._train_model(game_history, player_position, winner_position)


    def _train_model(self, game_history, player_position, winner_position):
        """
        Adjusts the number of beads in the matchboxes.

        Args:
             game_history (list): A list of (state, move) tuples representing the game history.
             player_position (int): The play order position (player number) of the MENACE agent.
             winner_position (int): The play order position (player number) of the winner of the game.
        """
        player_idx = player_position - 1

        if player_position == winner_position:
            for state, move in game_history[player_idx::2]:
                self.matchboxes[state].reward(move) # add beads
        else:
            for state, move in game_history[player_idx::2]:
                self.matchboxes[state].punish(move) # remove beads

        self.save_model()