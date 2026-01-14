import random
import math


# The aim of this coursework is to implement the minimax algorithm to determine the next move for a game of Connect.
# The goal in Connect is for a player to create a line of the specified number of pieces, either horizontally, vertically or diagonally.
# It is a 2-player game with each player having their own type of piece, "X" and "O" in this instantiation.
# You will implement the strategy for the first player, who plays "X". The opponent, who always goes second, plays "O".
# The number of rows and columns in the board varies, as does the number of pieces required in a line to win.
# Each turn, a player must select a column in which to place a piece. The piece then falls to the lowest unfilled location.
# Rows and columns are indexed from 0. Thus, if at the start of the game you choose column 2, your piece will fall to row 0 of column 2.
# If the opponent also selects column 2 their piece will end up in row 1 of column 2, and so on until column 2 is full (as determined
# by the number of rows).
# Note that board locations are indexed in the data structure as [row][column]. However, you should primarily be using checkFull(),
# checkSpace() etc. in board.py rather than interacting directly with the board.gameBoard structure.
# It is recommended that look at the comments in board.py to get a feel for how it is implemented.
#
# Your task is to complete the two methods, 'getMove()' and 'getMoveAlphaBeta()'.
#
# getMove() should implement the minimax algorithm, with no pruning. It should return a number, between 0 and (maxColumns - 1), to
# select which column your next piece should be placed in. Remember that columns are zero indexed, and so if there are 4 columns in
# you must return 0, 1, 2 or 3.
#
# getMoveAlphaBeta() should implement minimax with alpha-beta pruning. As before, it should return the column that your next
# piece should be placed in.
#
# The only imports permitted are those already imported. You may not use any additional resources. Doing so is likely to result in a
# mark of zero. Also note that this coursework is NOT an exercise in Python proficiency, which is to say you are not expected to use the
# most "Pythonic" way of doing things. Your implementation should be readable and commented appropriately. Similarly, the code you are
# given is intended to be readable rather than particularly efficient or "Pythonic".
#
# IMPORTANT: You MUST TRACK how many nodes you expand in your minimax and minimax with alpha-beta implementations.
# IMPORTANT: In your minimax with alpha-beta implementation, when pruning you MUST TRACK the number of times you prune.
class Player:
    # note that because of the game rule, the piece must fall into the lowest unfilled location. The user has no control over the row and only the column. Thus when we construct the algorithm, we only return a single interger within [0, maxColumns - 1]

    def __init__(self, name):
        self.name = name
        self.numExpanded = 0  # Use this to track the number of nodes you expand
        self.numPruned = 0  # Use this to track the number of times you prune

    def getMove(self, gameBoard) -> [int]:
        """
        minimax algorithm - without pruning
        returns the best column play base on input x
        and exploring all possible moves
        """
        # rest counter for the move because we are using the same object for both players (this algorithm work for both sides)
        self.numExpanded = 0

        # identify if we are trying to minimize or maximize the current status of the boar:
        opponent = "O" if self.name == "X" else "X"

        # initialize the best score and best move:
        bestScore = -math.inf
        bestMove = 0  # default to the first column, not setting this to -1 because we must made a decision regardless

        # iterate through all possible moves:
        for col in range(gameBoard.numColumns):
            # check if the column is full:
            # Check if the current column 'col' is not full.
            # gameBoard.colFills[col] is the number of pieces currently in column 'col'.
            # gameBoard.numRows is the total number of rows available in the board.
            if gameBoard.colFills[col] < gameBoard.numRows:
                # create a copy of the board and make a move on it:
                boardCopy = gameBoard.copy()
                boardCopy.addPiece(col, self.name)

                # recursively evaluate the score of the board (opponent's perspective):
                score = self._minimax(boardCopy, False, self.name, opponent)

                # update the best score and best move:
                if score > bestScore:
                    bestScore = score
                    bestMove = col

        # return the best move:
        return bestMove

    def getMoveAlphaBeta(self, gameBoard):
        """
        Minimax algorithm with alpha-beta pruning.
        Same as getMove() but uses pruning to skip branches that can't improve the result.
        """
        # Reset counters for this move
        self.numExpanded = 0
        self.numPruned = 0

        opponent = "O" if self.name == "X" else "X"

        bestScore = float("-inf")
        bestMove = 0

        # Alpha-beta bounds: alpha = best score for maximizer, beta = best score for minimizer
        alpha = float("-inf")
        beta = float("inf")

        # Try each possible column
        for col in range(gameBoard.numColumns):
            if gameBoard.colFills[col] < gameBoard.numRows:
                boardCopy = gameBoard.copy()
                boardCopy.addPiece(col, self.name)

                # Recursively evaluate with alpha-beta pruning
                score = self._minimaxAlphaBeta(
                    boardCopy, False, self.name, opponent, alpha, beta
                )

                # Update best move if this is better
                if score > bestScore:
                    bestScore = score
                    bestMove = col

                # Update alpha
                alpha = max(alpha, bestScore)

        return bestMove

    def _minimax(self, board, isMaximizing: bool, player, opponenet) -> int:
        """
                Recursive Minimax function without pruning:
        args:
            board
            ismaximizing: bool (true if its the maximizng player's turn, false is the minimizing player's turn)
            player: str
            opponenet: str
                Returns the best score for the current player based on the board state and the player's perspective.

        """
        # increment the number of nodes expanded:
        self.numExpanded += 1

        # check for terminal states:
        # base case: if the board is a win, return the score:
        if board.checkWin():
            # large magnitude ensures the terminal states dominated

            # board.lastPlay structure is: (row_index, column_index, player_name).
            # To decide who won, we check board.lastPlay[2].
            # If it equals 'player' (the maximizing player), it's a win for the maximizer.
            return 1000 if board.lastPlay[2] == player else -1000

        # base case: if the board is full, return 0:
        # because no one won, thus is a draw
        if board.checkFull():
            return 0

        if isMaximizing:
            bestScore = -math.inf

            # try all possible moves and return the best score:
            for col in range(board.numColumns):
                if board.colFills[col] < board.numRows:
                    # create a copy of the board and make a move on it:
                    boardCopy = board.copy()
                    boardCopy.addPiece(col, player)
                    # recursively evaluate the score of the board:
                    score = self._minimax(boardCopy, False, player, opponenet)
                    bestScore = max(bestScore, score)
            return bestScore

        else:
            bestScore = math.inf
            # try all possible moves and return the lowest score:

            for col in range(board.numColumns):
                if board.colFills[col] < board.numRows:
                    # create a copy of the board and make a move on it:
                    boardCopy = board.copy()
                    boardCopy.addPiece(col, opponenet)
                    # recursively evaluate the score of the board:
                    # After opponent moves, it becomes our turn (maximizing), so pass True
                    score = self._minimax(boardCopy, True, player, opponenet)
                    # track the worst score from our perspective (opponent wants to minimize our score):
                    bestScore = min(bestScore, score)
            return bestScore

    def _minimaxAlphaBeta(self, board, isMaximizing, player, opponent, alpha, beta):
        """
            Recursive minimax function with alpha-beta pruning.

            Args:
                    board
        ismaximizing: bool (true if its the maximizng player's turn, false is the minimizing player's turn)
        player: str
        opponenet: str
            Returns the best score for the current player based on the board state and the player's perspective.

        """
        # increment the number of nodes expanded:
        self.numExpanded += 1

        # Check terminal states (same as regular minimax)
        if board.checkWin():
            if board.lastPlay[2] == player:
                return 1000
            else:
                return -1000

        if board.checkFull():
            return 0

        # If maximizing (our turn)
        if isMaximizing:
            bestScore = float("-inf")

            for col in range(board.numColumns):
                if board.colFills[col] < board.numRows:
                    board.addPiece(col, player)
                    score = self._minimaxAlphaBeta(
                        board, False, player, opponent, alpha, beta
                    )
                    board.removePiece(col)

                    bestScore = max(bestScore, score)
                    alpha = max(alpha, bestScore)  # Update alpha

                    # PRUNING: If alpha >= beta, opponent won't choose this branch
                    # because they can get a better score elsewhere
                    if alpha >= beta:
                        self.numPruned += 1
                        break  # Skip remaining moves in this branch

            return bestScore

        # If minimizing
        else:
            worstScore = float("inf")

            for col in range(board.numColumns):
                if board.colFills[col] < board.numRows:
                    board.addPiece(col, opponent)
                    score = self._minimaxAlphaBeta(
                        board, True, player, opponent, alpha, beta
                    )
                    board.removePiece(col)

                    worstScore = min(worstScore, score)
                    beta = min(beta, worstScore)  # Update beta

                    # PRUNING: If beta <= alpha, we won't choose this branch
                    # because we can get a better score elsewhere
                    if beta <= alpha:
                        self.numPruned += 1
                        break  # Skip remaining moves in this branch

            return worstScore

            # since the connect 4 is
