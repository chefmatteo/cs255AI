import board


# Game class specifically designed for two bots to play against each other
# This replicates the structure of game.py but is optimized for bot vs bot gameplay
class BotsGame:
    # Store the players and create a board with the given specification
    def __init__(self, player1, player2, rows, columns, winNum):
        self.player1 = player1
        self.player2 = player2
        self.gameBoard = board.Board(rows, columns, winNum)
        self.listOfPlayers = (player1, player2)

    # Play the game itself, with or without alpha-beta pruning according to whether the pruning
    # argument is true or false respectively.
    # For bot vs bot games, both players can use alpha-beta if configured
    def playGame(self, player1_pruning=False, player2_pruning=False):
        """
        Play a game between two bots.

        Args:
            player1_pruning: If True, player 1 uses alpha-beta pruning
            player2_pruning: If True, player 2 uses alpha-beta pruning
        """
        # Keep track of whether the game is won or the board is full
        won = False
        full = False
        # index is used to keep track of which player's move it is
        index = 0
        # Display initial board
        print("\n=== Starting Bot vs Bot Game ===")
        print(
            f"Board size: {self.gameBoard.numRows} rows x {self.gameBoard.numColumns} columns"
        )
        print(f"Pieces needed to win: {self.gameBoard.winNum}")
        print(
            f"Player 1 ({self.player1.name}): {'Alpha-Beta' if player1_pruning else 'Minimax'}"
        )
        print(
            f"Player 2 ({self.player2.name}): {'Alpha-Beta' if player2_pruning else 'Minimax'}"
        )
        print("\nInitial board:")
        self.gameBoard.printBoard()
        print()
        # play the game until a player wins or it is a draw (i.e., the board is full)
        while not won and not full:
            # Get the current player, and update the index
            currPlayer = self.listOfPlayers[index]
            print("Moving with player " + str(currPlayer.name))

            # Determine which move method to use based on player and pruning settings
            use_alpha_beta = False
            if index == 0 and player1_pruning:
                use_alpha_beta = True
            elif index == 1 and player2_pruning:
                use_alpha_beta = True
            # Also check if player has use_alpha_beta attribute (for opponent.py)
            elif (
                index == 1
                and hasattr(currPlayer, "use_alpha_beta")
                and currPlayer.use_alpha_beta
            ):
                use_alpha_beta = True

            if use_alpha_beta:
                move = currPlayer.getMoveAlphaBeta(self.gameBoard.copy())
            else:
                move = currPlayer.getMove(self.gameBoard.copy())

            moveDone = self.gameBoard.addPiece(move, currPlayer.name)
            if moveDone == True:
                won = self.gameBoard.checkWin()
                full = self.gameBoard.checkFull()
                # Print board after each move to observe game progress
                print(f"Player {currPlayer.name} placed piece in column {move}")
                self.gameBoard.printBoard()
                print()  # Add blank line for readability
            else:
                print("Player made illegal move. Turn lost.")

            index = (index + 1) % 2

        # Game over - determine winner and show statistics
        if won and currPlayer == self.player1:
            print("\n=== Game Over ===")
            print(f"Player {self.player1.name} Wins!")
            print("\nPlayer 1 Statistics:")
            if hasattr(self.player1, "numExpanded"):
                print("  Nodes expanded:", self.player1.numExpanded)
                print("  Branches pruned:", self.player1.numPruned)
            print("\nPlayer 2 Statistics:")
            if hasattr(self.player2, "numExpanded"):
                print("  Nodes expanded:", self.player2.numExpanded)
                print("  Branches pruned:", self.player2.numPruned)
            print("\nFinal board:")
            self.gameBoard.printBoard()
            return 1

        if won and currPlayer == self.player2:
            print("\n=== Game Over ===")
            print(f"Player {self.player2.name} Wins!")
            print("\nPlayer 1 Statistics:")
            if hasattr(self.player1, "numExpanded"):
                print("  Nodes expanded:", self.player1.numExpanded)
                print("  Branches pruned:", self.player1.numPruned)
            print("\nPlayer 2 Statistics:")
            if hasattr(self.player2, "numExpanded"):
                print("  Nodes expanded:", self.player2.numExpanded)
                print("  Branches pruned:", self.player2.numPruned)
            print("\nFinal board:")
            self.gameBoard.printBoard()
            return -1

        if full and not won:
            print("\n=== Game Over ===")
            print("It's a Draw!")
            print("\nPlayer 1 Statistics:")
            if hasattr(self.player1, "numExpanded"):
                print("  Nodes expanded:", self.player1.numExpanded)
                print("  Branches pruned:", self.player1.numPruned)
            print("\nPlayer 2 Statistics:")
            if hasattr(self.player2, "numExpanded"):
                print("  Nodes expanded:", self.player2.numExpanded)
                print("  Branches pruned:", self.player2.numPruned)
            print("\nFinal board:")
            self.gameBoard.printBoard()
            return 0
