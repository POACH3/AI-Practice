"""
menace.py

Implements Donald Michie's MENACE (Machine Educable Noughts And Crosses Engine).

A reinforcement learning algorithm for small, discrete state spaces.
The entire statespace must be calculated in advance. The possible moves at
each state are associated with a probability distribution that is updated over
repeated iterations based on outcomes.

NOTES:
    board_states and moves should be generic and internally convert to string for saving
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
            self.model_path = f'{self.game_name}_player{self.player_position}_menace_model.json' # load default model for that game

        if os.path.exists(self.model_path):
            self.import_model(self.model_path)
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

            matchbox = Matchbox(state, moves_and_beads)
            self.matchboxes[state] = matchbox

        self.model_path = f'{game_name}_player{player_position}_menace_model.json'

        model = {}
        for state, matchbox in self.matchboxes.items():
            model[state] = matchbox.moves_and_beads

        with open(self.model_path, 'w') as f:
            json.dump(model, f, indent=4)


    def import_model(self, model_file):
        """
        Reads in a JSON file that represents the states and beads and initializes
        all the matchboxes for the game.

        Args:
            model_file (string): Path to the JSON file.
        """
        with open(model_file, 'r') as f:
            model = json.load(f)

        for state, moves_and_beads in model.items():
            self.matchboxes[state] = Matchbox(state, moves_and_beads)


    def save_model(self):
        """
        Saves the model as JSON, writing over the original.
        """
        model = {}
        for state, matchbox in self.matchboxes.items():
            model[state] = matchbox.moves_and_beads

        with open(self.model_path, 'w') as f:
            json.dump(model, f, indent=4)


    def get_move(self, board_state):
        """
        Gets the move to make, given the board state.

        Args:
            board_state (string): A string representing the board state.

        Returns:
            move: A representation of the move to make.
        """
        matchbox = self.matchboxes.get(board_state)

        if matchbox is None:
            raise Exception(f'No matchbox found for board state: {board_state}')

        move = matchbox.select_bead()
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