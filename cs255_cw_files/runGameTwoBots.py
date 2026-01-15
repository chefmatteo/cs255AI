import bots_game
import player
import opponent

# Example script showing how to have two bots play against each other
# Uses bots_game.py which is specifically designed for bot vs bot gameplay

# Option 1: Both players using alpha-beta pruning (fastest)
p1 = player.Player("X")
p2 = opponent.Opponent("O", use_alpha_beta=True)

# Option 2: Both players using regular minimax (no pruning)
# p1 = player.Player("X")
# p2 = player.Player("O")

# Option 3: One with alpha-beta, one without
# p1 = player.Player("X")
# p2 = player.Player("O")

# Create game with two bots
# Arguments: player1, player2, rows, columns, winNum
# g = bots_game.BotsGame(p1, p2, 4, 4, 4)
# g = bots_game.BotsGame(p1, p2, 5, 6, 3)
# g = bots_game.BotsGame(p1, p2, 4, 5, 3)
# g = bots_game.BotsGame(p1, p2, 4, 4, 4)
# g = bots_game.BotsGame(p1, p2, 4, 4, 3)
g = bots_game.BotsGame(p1, p2, 8, 8, 3)

# Play game with independent pruning control for each player
# Arguments: player1_pruning, player2_pruning
# Both True = both use alpha-beta (recommended for performance)
g.playGame(player1_pruning=True, player2_pruning=True)
