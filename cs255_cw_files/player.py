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
    # note that because of the game rule, the piece must fall into the lowest unfilled location. The user has no control over the row and only the column. Thus when we construct the algorithm, we only return a single integer within [0, maxColumns - 1]

    def __init__(self, name):
        self.name = name
        self.numExpanded = 0  # Use this to track the number of nodes you expand
        self.numPruned = 0  # Use this to track the number of times you prune

    def _calculateMaxDepth(self, board):
        """
        Calculate adaptive maximum depth based on board size and game complexity.
        
        arguments:
            board: The game board state
            
        expected return:
            int: A reasonable depth limit that balances search quality vs. computation time.
        """
        # Notice that depth is important for the performance of the algorithm, because it limits the number of nodes that are explored, for larger boards, we are looking at an astronomical number of nodes to explore
        # In this project we assume the board size is within a considerable amount, which is [4,4] to [12,12]. We hardcoded the depth search limit for the project in such case to make sure the computation time is reasonable. 
        
        # logic: 
        # Calculate adaptive maximum depth based on board size and game complexity
        # Returns a reasonable depth limit that balances search quality vs. computation time.

        # Base depth calculation: consider board size and win requirement
        totalSpaces = board.numRows * board.numColumns

        # Logic:
        # For small boards (e.g., 4x4), we can search deeper
        # For large boards (e.g., 8x8), we need shallower search
        if totalSpaces <= 16:  # 4x4 or smaller
            maxDepth = 8
        elif totalSpaces <= 30:  # 5x6 or 6x5
            maxDepth = 6
        elif totalSpaces <= 42:  # 6x7 (standard Connect 4)
            maxDepth = 5
        elif totalSpaces <= 64:  # 8x8
            maxDepth = 4
        else:
            maxDepth = 3

        # Adjust based on win requirement: smaller winNum means longer games (more defense and less aggressive moves), need shallower depth
        if board.winNum <= 2:
            # For very small win requirements on large boards, use very shallow depth
            if totalSpaces >= 64:  # Large boards (8x8 or bigger)
                maxDepth = 2  # Very shallow for large boards with small winNum
            else:
                maxDepth = max(2, maxDepth - 2)
        elif board.winNum == 3:
            maxDepth = max(3, maxDepth - 1)  # Slightly reduce depth

        return maxDepth


    # Adaptive depth is used to set the maximum search depth for the minimax algorithm dynamically,
    # Rather than using a fixed value, such allowed search depth (maxDepth) is computed based on board size
    # and win condition. On smaller boards or with higher win requirements, a deeper search is feasible and useful, which ensure the model explore all the possible moves and make the best decision, thus making the AI stronger. 
    # On large boards or with small win requirements, searching too deeply would be computationally expensive or unnecessary, so the depth is reduced.
   
    def getMove(self, gameBoard) -> [int]:
        """
        minimax algorithm - without pruning
        returns the best column play base on input x
        and exploring all possible moves
        
        arguments: 
            gameBoard: The current game board state
            
        expected return: 
            int: The column index (0 to numColumns-1) representing the best move
        """
        # expanded -> track the number of nodes that are explored
        self.numExpanded = 0

        # identify if we are trying to minimize or maximize the current status of the board:
        opponent = "O" if self.name == "X" else "X"

        # initialize the best score and best move:
        bestScore = -math.inf
        bestMove = 0  # default to the first column, not setting this to -1 because we must made a decision regardless

        # Calculate adaptive depth limit based on board size
        maxDepth = self._calculateMaxDepth(gameBoard)

        # iterate through all possible moves:
        for col in range(gameBoard.numColumns):
            # check if the column is full:
            if gameBoard.colFills[col] < gameBoard.numRows:
                # create a copy of the board and make a move on it:
                # Avoid modifying the original board 
                boardCopy = gameBoard.copy()
                boardCopy.addPiece(col, self.name)

                # recursively evaluate the score of the board (opponent's perspective):
                # Pass the adaptive depth limit
                score = self._minimax(boardCopy, False, self.name, opponent, maxDepth)

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
        
        arguments: 
            gameBoard: The current game board state
            
        expected return: 
            int: The column index (0 to numColumns-1) representing the best move
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

        # Calculate adaptive depth limit based on board size
        maxDepth = self._calculateMaxDepth(gameBoard)

        # Try each possible column
        for col in range(gameBoard.numColumns):
            if gameBoard.colFills[col] < gameBoard.numRows:
                boardCopy = gameBoard.copy()
                boardCopy.addPiece(col, self.name)

                # Recursively evaluate with alpha-beta pruning
                # Pass the adaptive depth limit and current alpha/beta bounds
                score = self._minimaxAlphaBeta(
                    boardCopy, False, self.name, opponent, alpha, beta, maxDepth
                )

                # Update best move if this is better
                if score > bestScore:
                    bestScore = score
                    bestMove = col

                # Update alpha (best score found so far for maximizer)
                alpha = max(alpha, bestScore)

                # Pruning: if alpha >= beta, we can skip remaining moves
                # (opponent won't allow us to get a better score)
                if alpha >= beta:
                    break

        return bestMove

    def _minimax(
        self, board, isMaximizing: bool, player, opponent, max_depth: int = 1000
    ) -> int:
        """
        Recursive Minimax function without pruning:
        arguments:
            board
            isMaximizing: bool (true if its the maximizing player's turn, false is the minimizing player's turn)
            player: str
            opponent: str
            
        Returns the best score for the current player based on the board state and the player's perspective.
        """
        # increment the number of nodes expanded:
        self.numExpanded += 1

        # check for terminal states FIRST (these are definitive):
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

        # check if we have reached the maximum depth (after checking terminal states):
        if max_depth <= 0:
            return self._evaluateBoard(board, player, opponent)

        if isMaximizing:
            bestScore = -math.inf

            # try all possible moves and return the best score:
            for col in range(board.numColumns):
                if board.colFills[col] < board.numRows:
                    # create a copy of the board and make a move on it:
                    boardCopy = board.copy()
                    boardCopy.addPiece(col, player)
                    # recursively evaluate the score of the board:
                    # Decrement depth for recursive call
                    score = self._minimax(
                        boardCopy, False, player, opponent, max_depth - 1
                    )
                    bestScore = max(bestScore, score)
            return bestScore

        else:
            bestScore = math.inf
            # try all possible moves and return the lowest score:

            for col in range(board.numColumns):
                if board.colFills[col] < board.numRows:
                    # create a copy of the board and make a move on it:
                    boardCopy = board.copy()
                    boardCopy.addPiece(col, opponent)
                    # recursively evaluate the score of the board:
                    # After opponent moves, it becomes our turn (maximizing), so pass True
                    # Decrement depth for recursive call
                    score = self._minimax(
                        boardCopy, True, player, opponent, max_depth - 1
                    )
                    # track the worst score from our perspective (opponent wants to minimize our score):
                    bestScore = min(bestScore, score)
            return bestScore

    def _minimaxAlphaBeta(
        self, board, isMaximizing, player, opponent, alpha, beta, max_depth: int = 1000
    ):
        """
        Recursive minimax function with alpha-beta pruning.

        arguments:
        board
        isMaximizing: bool (true if its the maximizing player's turn, false is the minimizing player's turn)
        player: str
        opponent: str
        
        expected return:    
        Returns the best score for the current player based on the board state and the player's perspective.

        """
        # increment the number of nodes expanded:
        self.numExpanded += 1

        # Check terminal states first because they are definitive (known state)
        if board.checkWin():
            if board.lastPlay[2] == player:
                return 1000
            else:
                return -1000

        if board.checkFull():
            return 0

        # check if we have reached the maximum depth (after checking terminal states):
        if max_depth <= 0:
            return self._evaluateBoard(board, player, opponent)

        # If maximizing (our turn)
        if isMaximizing:
            bestScore = float("-inf")

            for col in range(board.numColumns):
                if board.colFills[col] < board.numRows:
                    board.addPiece(col, player)
                    score = self._minimaxAlphaBeta(
                        board, False, player, opponent, alpha, beta, max_depth - 1
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
                        board, True, player, opponent, alpha, beta, max_depth - 1
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




    
    # BestScore:
    # bestScore in getMove() -> used to find the best column to play at the root level
    # used to compare scores across different columns
    # e.g the column 10 has a score of 100, and the column 11 has a score of 90, then the best score is 100. We constantly iterate all possible moves and pick the best one. 
        
    # bestScore in _minimax() -> keeps track of the best score found at a given node in the game tree
    # compares scores across all possible moves at that node
    # in a search tree bestScore refers to the optimal score found at the root node
    
    # Note that minimax with alpha-beta algorithm must reach the end of the tree to return the best score via backtracking, it obviously doesn't make any sense for it to search the board for all possible moves if e.g. num_rows = 8, num_columns = 8, and win_num = 2, because the algorithm will have to search 8^8 = 16777216 nodes, which is obviously not feasible.

    # Even with the use of alpha-beta pruning, the searching over 8^64 nodes is still an astronomical amount to the computer, thus there is a necessity to implement a heuristic function to guide the search, and a depth limiting function to limit the search to a reasonable depth.

    # But we only use such methodology in a more high number of possibilities, when is feasible to reach all the terminal states, we prefer to use minimax

    # Thus in future work, we can implement some advanced heuristics to guide the search, such as threat detection, center control (widely adopted in chess) and weighted pattern scoring to make the AI stronger.
    
    

    def _evaluateBoard(self, board, player, opponent):
        """
        Evaluate board position without reaching terminal state.
        Returns a score estimate from player's perspective.
        
        Considers:
        1. Piece count difference (base score)
        2. Center control (center columns are more valuable)
        """
        score = 0
        centerCol = board.numColumns // 2
        
        # Evaluate each position: piece count and center control
        for row in range(board.numRows):
            for col in range(board.numColumns):
                space = board.checkSpace(row, col)
                
                if space.value == player:
                    # Base piece value
                    score += 1
                    # Center control bonus: pieces in center columns are more valuable
                    distanceFromCenter = abs(col - centerCol)
                    centerBonus = (board.numColumns // 2) - distanceFromCenter
                    score += centerBonus
                    
                elif space.value == opponent:
                    # Base piece penalty
                    score -= 1
                    # Penalize opponent's center control
                    distanceFromCenter = abs(col - centerCol)
                    centerPenalty = (board.numColumns // 2) - distanceFromCenter
                    score -= centerPenalty
        
        return score
    
