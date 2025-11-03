"""
menace.py

Implements Donald Michie's MENACE (Machine Educable Noughts And Crosses Engine).

A reinforcement learning algorithm for small, discrete state spaces.
The entire statespace must be calculated in advance. The possible moves at
each state are associated with a probability distribution that is updated over
repeated iterations based on outcomes.
"""

import csv

class Menace:
    """
    Represents a MENACE agent that learns optimal moves through reinforcement.
    """

    def __init__(self, model_csv):
        # needs a list of the states
        self.matchboxes = {}
        self.import_model(model_csv)

        self.game_history = [] # (state, move) tuples


    def initialize_model(self, model_csv):
        """
        Generates game states.
        Then reads in the .csv file created.

        that represents the states and beads and initializes all the matchboxes for the game.

        Args:
            model_csv (string): Path to the .csv file.
        """
        pass

    def import_model(self, model_csv):
        """
        Reads in a .csv file that represents the states and beads and initializes
        all the matchboxes for the game.

        Args:
            model_csv (string): Path to the .csv file.
        """
        pass


    def get_move(self, board_state):
        """
        Gets the move to make, given the board state.

        Args:
            board_state (string): A string representing the board state.

        Returns:
            move: A representation of the move to make.
        """
        move = self.matchboxes[board_state].select_bead()
        self.game_history.append((board_state, move))
        return move


    def train_model(self, is_win):
        """
        Adjusts the number of beads in the matchboxes.

        Args:
             is_win (bool): True if the game was won, False if lost.
        """
        if is_win:
            for state in self.game_history:
                self.matchboxes[state].set_beads() # add
        else:
            for state in self.game_history:
                self.matchboxes[state].set_beads() # remove


    def save_model(self, model_path):
        pass