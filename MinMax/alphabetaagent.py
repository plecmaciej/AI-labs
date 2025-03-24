from exceptions import GameplayException
from connect4 import Connect4
class AlphaBetaAgent:
    def __init__(self, my_token):
        self.my_token = my_token

    def decide(self, connect4):
        try:
            _, column = self.alphabeta(connect4, 4, float('-inf'), float('inf'), True)
            return column
        except GameplayException:
            # Jeśli gra jest już zakończona, agent nie podejmuje decyzji.
            return None

    def alphabeta(self, connect4, depth, alpha, beta, maximizing_player):
        if depth == 0 or connect4.game_over:
            return self.evaluate(connect4), None

        if maximizing_player:
            max_eval = float('-inf')
            best_column = None
            for column in connect4.possible_drops():
                new_connect4 = self.simulate_drop(connect4, column)
                eval, _ = self.alphabeta(new_connect4, depth - 1, alpha, beta, False)
                if eval > max_eval:
                    max_eval = eval
                    best_column = column
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # odcięcie beta
            return max_eval, best_column
        else:
            min_eval = float('inf')
            best_column = None
            for column in connect4.possible_drops():
                new_connect4 = self.simulate_drop(connect4, column)
                eval, _ = self.alphabeta(new_connect4, depth - 1, alpha, beta, True)
                if eval < min_eval:
                    min_eval = eval
                    best_column = column
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # odcięcie alfa
            return min_eval, best_column

    def evaluate(self, connect4):
        if connect4.wins == self.my_token:
            return 1
        elif connect4.wins is None:
            return 0
        else:
            return -1

    def simulate_drop(self, connect4, column):
        new_connect4 = Connect4(connect4.width, connect4.height)
        new_connect4.board = [row[:] for row in connect4.board]
        new_connect4.who_moves = connect4.who_moves
        new_connect4.game_over = connect4.game_over
        new_connect4.wins = connect4.wins

        new_connect4.drop_token(column)
        return new_connect4
