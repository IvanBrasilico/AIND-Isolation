"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
from operator import itemgetter


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


class MaxDepthAchieved(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # Returns actual player number of moves minus opponent number of moves
    # divided by distance of center. 
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    my_moves = len(game.get_legal_moves())
    op_moves = len(game.get_legal_moves(game.get_opponent(player)))
    w, h = game.width / 2., game.height / 2.
    y, x = game.get_player_location(player)
    dist_x = abs(h - x)
    dist_y = abs(w - y)
    return float((my_moves - op_moves) / ((dist_x + dist_y) * 2))


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # Returns opponent number of moves minus actual player number of moves
    # divided by distance of oponent. 
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    my_moves = len(game.get_legal_moves())
    op_moves = len(game.get_legal_moves(game.get_opponent(player)))
    w, h = game.get_player_location(game.get_opponent(player))
    y, x = game.get_player_location(player)
    dist_x = abs(h - x)
    dist_y = abs(w - y)
    return float((my_moves - op_moves) / ((dist_x + dist_y) * 2))


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if len(game.get_blank_spaces()) / (game.width * game.height) > 0.2:
        return custom_score(game, player)
    return custom_score_2(game, player)

class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.

    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)
        """actual_moves = []
        for move in game.get_legal_moves():
            new_game = game.forecast_move(move)
            score = self.score(new_game, new_game.active_player)
            actual_moves.append((move, score))
        if actual_moves:
            actual_moves.sort(key=itemgetter(1))
            return actual_moves[-1][0]
        """
        try:
            return self.minimax(game, self.search_depth)
        except SearchTimeout:
            # print("TIMEOUT!!")
            pass

        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        bestmove = (-1, -1)
        bestscore = float('-inf')
        legal_moves = game.get_legal_moves()
        if len(legal_moves) == 0:
            return (-1, -1)
        for move in legal_moves:
            value = self.min_value(game.forecast_move(move), depth - 1)
            if value > bestscore:
                bestscore = value
                bestmove = move
        return bestmove

    def terminal_test(self, game, depth):
        """ Return True if the game is over for the active player
        and False otherwise.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # If max_depth or timeout is achieved on next iters, save last
        # level searched to evaluate (ID)
        self.last_level = game
        if depth <= 0:
            return True

        return not bool(game.get_legal_moves())

    def min_value(self, game, depth):
        """ Return the value for a win (+1) if the game is over,
        otherwise return the minimum value over all legal child
        nodes.
        """
        if self.terminal_test(game, depth):
            return self.score(game, game.active_player)

        # If max_depth or timeout is achieved on next iters, save last
        # level searched to evaluate (ID)
        self.last_level = game

        v = float("inf")
        for m in game.get_legal_moves():
            v = min(v, self.max_value(game.forecast_move(m), depth - 1))
        return v

    def max_value(self, game, depth):
        """ Return the value for a loss (-1) if the game is over,
        otherwise return the maximum value over all legal child
        nodes.
        """
        if self.terminal_test(game, depth):
            return -self.score(game, game.active_player)

        # If max_depth or timeout is achieved on next iters, save last
        # level searched to evaluate (ID)
        self.last_level = game

        v = float("-inf")
        for m in game.get_legal_moves():
            v = max(v, self.min_value(game.forecast_move(m), depth - 1))
        return v


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left
        depth = 1
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            while True:
                best_move = self.alphabeta(game, depth)
                depth +=1
        except SearchTimeout:
           pass

        return best_move



    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        bestscore = float('-inf')
        bestmove = None
        legal_moves = game.get_legal_moves()
        if len(legal_moves) == 0:
            return (-1, -1)
        for move in legal_moves:
            value = self.min_value(game.forecast_move(move), alpha, beta, depth - 1)
            if value > bestscore:
                bestscore = value
                bestmove = move
        return bestmove

    def terminal_test(self, game, depth):
        """ Return True if the game is over for the active player
        and False otherwise.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # If max_depth or timeout is achieved on next iters, save last
        # level searched to evaluate (ID)
        self.last_level = game

        if depth <= 0:
            return True

        return not bool(game.get_legal_moves())

    def min_value(self, game, alpha, beta, depth):
        """ Return the value for a win (+1) if the game is over,
        otherwise return the minimum value over all legal child
        nodes.
        """
        if self.terminal_test(game, depth):
            return self.score(game, game.active_player)

        # If max_depth or timeout is achieved on next iters, save last
        # level searched to evaluate (ID)
        self.last_level = game

        v = float("inf")
        for m in game.get_legal_moves():
            v = min(v, self.max_value(game.forecast_move(m), alpha, beta, depth - 1))
            if v <= alpha:
                return v
            alpha = min(alpha, v)
        return v

    def max_value(self, game, alpha, beta, depth):
        """ Return the value for a loss (-1) if the game is over,
        otherwise return the maximum value over all legal child
        nodes.
        """
        if self.terminal_test(game, depth):
            return -self.score(game, game.active_player)

        v = float("-inf")
        for m in game.get_legal_moves():
            v = max(v, self.min_value(game.forecast_move(m), alpha, beta, depth - 1))
            if v >= beta:
                return v
            beta = min(beta, v)
        return v

class AlphaBetaLastLevelPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. 
    Do AlphaBeta, saving last level achieved.
    If max_depth or timeout, evaluate heuristic score of last level achieved.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        best_move = (-1, -1)
        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            best_move = self.alphabeta(game, self.search_depth)
        except (SearchTimeout, MaxDepthAchieved):
            # If there are remaining moves, use evaluation function
            actual_moves = []
            for move in self.last_level.get_legal_moves():
                new_game = self.last_level.forecast_move(move)
                score = self.score(new_game, self.last_level.active_player)
                actual_moves.append((move, score))
            if actual_moves:
                actual_moves.sort(key=itemgetter(1))
                if game.active_player == self.last_level.active_player:
                    return actual_moves[-1][0]
                else:
                    return actual_moves[0][0]
                    

        return best_move



    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        bestscore = float('-inf')
        bestmove = None
        legal_moves = game.get_legal_moves()
        if len(legal_moves) == 0:
            return (-1, -1)
        for move in legal_moves:
            value = self.min_value(game.forecast_move(move), alpha, beta, depth - 1)
            if value > bestscore:
                bestscore = value
                bestmove = move
        return bestmove

    def terminal_test(self, game, depth):
        """ Return True if the game is over for the active player
        and False otherwise.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if depth <= 0:
            raise MaxDepthAchieved()

        # If max_depth or timeout is achieved on next iters, save last
        # level searched to evaluate (ID)
        self.last_level = game

        return not bool(game.get_legal_moves())

    def min_value(self, game, alpha, beta, depth):
        """ Return the value for a win (+1) if the game is over,
        otherwise return the minimum value over all legal child
        nodes.
        """
        if self.terminal_test(game, depth):
            return 1

        # If max_depth or timeout is achieved on next iters, save last
        # level searched to evaluate (ID)
        self.last_level = game

        v = float("inf")
        for m in game.get_legal_moves():
            v = min(v, self.max_value(game.forecast_move(m), alpha, beta, depth - 1))
            if v <= alpha:
                return v
            alpha = min(alpha, v)
        return v

    def max_value(self, game, alpha, beta, depth):
        """ Return the value for a loss (-1) if the game is over,
        otherwise return the maximum value over all legal child
        nodes.
        """
        if self.terminal_test(game, depth):
            return -1

        v = float("-inf")
        for m in game.get_legal_moves():
            v = max(v, self.min_value(game.forecast_move(m), alpha, beta, depth))
            if v >= beta:
                return v
            beta = min(beta, v)
        return v
