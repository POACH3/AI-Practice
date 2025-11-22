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
    Represents a MENACE matchbox, a structure to hold learned move probabilities.
    A matchbox is associated with a single game state and holds a probability distribution over all the
    legal moves that can be reached from that state. This probability distribution over moves is represented by
    a certain number of beads which are associated with each move.
    """

    def __init__(self, state, moves_and_beads):
        """
        Constructor.

        Args:
            state (str): The game state this matchbox represents.
            moves_and_beads (dict): Maps moves legally reached from this game state to number of beads associated with that move.
        """
        self.state = state
        self.moves_and_beads = moves_and_beads


    def get_beads(self, move):
        """
        Gets the number of beads associated with a move. The beads represent
        the probability of that move being selected.

        Args:
            move: The move.

        Returns:
            (int): The number of beads associated with the move.
        """
        return self.moves_and_beads.get(move, -1)


    def _set_beads(self, move, change):
        """
        Sets the number of beads associated with a move. This increases or decreases
        the probability of the move being selected.
        The number of beads associated with the move is clamped to ensure that there
        remains a probability distribution.

        Args:
            move: The move associated with the number of beads.
            change (int): The number of beads to be added or removed.
        """
        min_beads = 1
        max_beads = 15

        self.moves_and_beads[move] = min(max(min_beads, self.moves_and_beads[move] + change), max_beads)


    def reward(self, move):
        """
        Rewards a model for a good move.
        Adds one bead to the matchbox representing that move.

        Args:
            move: The move to be rewarded.
        """
        self._set_beads(move, 1)


    def punish(self, move):
        """
        Punishes a model for a bad move.
        Unless there is only one bead left, removes one bead from the matchbox representing that move.

        Args:
            move: The move to be punished.
        """
        self._set_beads(move, -1)


    def draw_bead(self):
        """
        Draws a random bead from the matchbox to select a move.

        Returns:
            move: The selected move.
        """
        moves = list(self.moves_and_beads.keys())
        weights = list(self.moves_and_beads.values())

        return random.choices(moves, weights=weights, k=1)[0]


    def to_dict(self):
        """
        Converts the matchbox to a dictionary.

        Returns:
            d (dict): The dictionary representation of the matchbox.
        """
        d = {
            'state': self.state,
            'moves': self.moves_and_beads,
        }

        return d


    @classmethod
    def from_dict(cls, d):
        """
        Converts a dictionary representation of a matchbox to a matchbox.

        Args:
            d (dict): The dictionary representation of a matchbox.

        Returns:
            (Matchbox): A matchbox.
        """
        return cls(d['state'], d['moves_and_beads'])