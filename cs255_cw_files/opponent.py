import player

# Opponent class that uses minimax algorithm to play optimally

class Opponent(player.Player):
    """
    An opponent player that uses minimax algorithm (same as Player class).
    This class inherits from Player to reuse the minimax implementation.

    Can be configured to use alpha-beta pruning or regular minimax.
    When used with game.py, if use_alpha_beta=True, the game will automatically
    use getMoveAlphaBeta() for better performance.

    Example usage:
        p1 = player.Player("X")
        p2 = opponent.Opponent("O", use_alpha_beta=True)
        g = game.Game(p1, p2, 4, 4, 4)
        g.playGame(True)  # Both players will use alpha-beta
    """

    def __init__(self, name, use_alpha_beta=False):
        """
        Initialize the opponent.

        Args:
            name: Player name ("X" or "O")
            use_alpha_beta: bool
        """
        super().__init__(name)
        self.use_alpha_beta = use_alpha_beta

    def getMove(self, gameBoard):
        if self.use_alpha_beta:
            return self.getMoveAlphaBeta(gameBoard)
        else:
            return super().getMove(gameBoard)
