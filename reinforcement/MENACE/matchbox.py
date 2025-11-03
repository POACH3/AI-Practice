"""
matchbox.py

Implements a structure analogous to Donald Michie's MENACE (Machine Educable
Noughts And Crosses Engine) matchboxes.
It is used to hold learned move probabilities in a generic
statespace with discrete moves (e.g. a chessboard).

NOTES:
    possibly extend functionality by creating a Bead class (do we need something more than a dictionary
        to encapsulate the information held by a "bead"?)
"""

import random

class Matchbox:
    """
    Represents a MENACE matchbox. A structure to hold learned move probabilities.
    """

    def __init__(self, state, moves, initial_beads=3):
        self.state = state
        self.beads = {move: initial_beads for move in moves}


    def get_beads(self, move):
        """
        Gets the number of beads associated with a move. The beads represent
        the probability of that move being selected.

        Args:
            move (string): A string representing the move.

        Returns:
            (int): The number of beads associated with the move.
        """
        if move in self.beads:
            return self.beads[move]
        else:
            return -1


    def _set_beads(self, move, change):
        """
        Sets the number of beads associated with a move. This increases or decreases
        the probability of the move being selected.

        Args:
            move (string): A string representing the move.
            change (int): The number of beads to be added or removed.
        """
        if move in self.beads:
            self.beads[move] = max(1, self.beads[move] + change)

        # do some sort of error checking?


    def reward(self, move):
        self._set_beads(move, 1)


    def punish(self, move):
        self._set_beads(move, -1)


    def select_bead(self):
        """
        Chooses a random bead from the matchbox.

        Returns:
             move (string): A string representing the move.
        """
        moves = list(self.beads.keys())
        weights = list(self.beads.values())
        move = random.choices(moves, weights=weights, k=1)[0]

        return move