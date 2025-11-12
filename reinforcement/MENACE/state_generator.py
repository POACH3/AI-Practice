"""
state_generator.py

Generates possible game states. MENACE must know all states in advance in
order to initialize the matchboxes for each state.
"""

import csv
import sys
import os


# HEXAPAWN

hexapawn_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../Hexapawn'))
sys.path.append(hexapawn_path)

from board import Board

# generate the states and legal moves (and initialize beads?)

def generate_hexapawn(board, player, all_states=None):
    if all_states is None:
        all_states = set()

    legal_moves = board.get_legal_moves(player)

    if len(legal_moves) != 0:
        for move in legal_moves:
            next_board = board.copy()
            next_board.move_piece(*move)

            if next_board.to_string() not in all_states:
                all_states.add(next_board.to_string())

                # change player
                next_player = 2 if player == 1 else 1

                # go to each new state, but check if win before trying to go to new state
                generate_hexapawn(next_board, next_player, all_states)

    return all_states


# TIC-TAC-TOE


def main():
    board = Board()
    player = 1

    game_states = generate_hexapawn(board, player)

    # write to CSV
    with open("hexapawn_model.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # write each board string as a row
        for state_str in sorted(game_states):
            writer.writerow([state_str])

if __name__ == "__main__":
    main()