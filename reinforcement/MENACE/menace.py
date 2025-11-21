"""
menace.py

Implements Donald Michie's MENACE (Machine Educable Noughts And Crosses Engine).

A reinforcement learning algorithm for small, discrete state spaces.
The entire statespace must be calculated in advance. The possible moves at
each state are associated with a probability distribution that is updated over
repeated iterations based on outcomes.

NOTES:
    consider json instead of a csv to hold the model
    board_states and moves should be generic and internally convert to string for saving
"""
import os
import csv
from matchbox import Matchbox


class Menace:
    """
    Represents a MENACE agent that learns optimal moves through reinforcement.
    """

    def __init__(self, player_position, game_name, states_and_moves=None, model_path=None):
        """
        Constructor.

        Args:
            player_position (int): The play order position (player number) of the MENACE agent.
            game_name (string): The name of the game being played.
            states_and_moves (dict): Maps game states to their legal moves. Optional if model already exists.
            model_path (string): The path to the model (a .csv file). Optional if default model exists or states_and_moves is not None.
        """
        self.game_type = game_name
        self.model_path = model_path
        self.player_position = player_position
        self.matchboxes = {}

        if self.model_path == None:
            self.model_path = game_name + '_player' + str(player_position) + '_menace_model.csv' # load default model for that game
            # add more game types here if necessary

        if not os.path.exists(model_path):
            raise FileNotFoundError('A model for this game was not found.') #FIXME - maybe instead initialize a new model
            #self.initialize_model(game_name, states_and_moves, player_position)
            #self.import_model(self.model_path)
        else:
            self.import_model(model_path)


    def initialize_model(self, player_position, game_name, states_and_moves, initial_beads=3):
        """
        Generates game states.
        Then reads in the .csv file created.

        that represents the states and beads and initializes all the matchboxes for the game.

        Args:
            player_position (int): The play order position (player number) of the MENACE agent.
            game_name (string): The name of the game being played.
            states_and_moves (dict): Maps game states to their legal moves.
            initial_beads (int): The number of initial beads to use for each move.
        """
        pass

        """
        self.model_path = game_name + '_player' + str(player_position) + 'menace_model.csv'

        with open(self.model_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # write each board string as a row
            for state_str in sorted(game_states):
                writer.writerow([state_str, initial_beads])
        """


    def import_model(self, model_csv):
        """
        Reads in a .csv file that represents the states and beads and initializes
        all the matchboxes for the game.

        Args:
            model_csv (string): Path to the .csv file.
        """
        with open(model_csv, newline='') as csvfile:
            reader = csv.reader(csvfile)

            for row in reader:
                matchbox = Matchbox(row[0], 3 if row[1] == None else row[1])
                self.matchboxes[row[0]] = matchbox


    def get_move(self, board_state):
        """
        Gets the move to make, given the board state.

        Args:
            board_state (string): A string representing the board state.

        Returns:
            move: A representation of the move to make.
        """
        matchbox = self.matchboxes.get(board_state)

        if matchbox == None:
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


    def save_model(self):
        """
        Saves the model, writing over the original.
        """
        with open(self.model_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            for matchbox in self.matchboxes.keys():
                writer.writerow([matchbox.state, matchbox.beads]) #FIXME - need to change so each row is state, move, beads, move, beads...