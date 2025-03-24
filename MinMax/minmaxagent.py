from exceptions import GameplayException
from connect4 import Connect4

class MinMaxAgent:
    def __init__(self, my_token):
        self.my_token = my_token
        self.last_heuristic_score = None
        self.isHeuristic = True

    def decide(self, connect4):
        try:
            _, column = self.minimax(connect4, 3, True)
            return column
        except GameplayException:
            # Jeśli gra jest już zakończona, agent nie podejmuje decyzji.
            return None

    def minimax(self, connect4, depth, maximizing_player):
        if depth == 0 or connect4.game_over:
            return self.evaluate(connect4), None

        if maximizing_player:
            max_eval = float('-inf')
            best_column = None
            for column in connect4.possible_drops():
                new_connect4 = self.simulate_drop(connect4, column)
                eval, _ = self.minimax(new_connect4, depth - 1, False)
                if eval > max_eval:
                    max_eval = eval
                    best_column = column
            return max_eval, best_column
        else:
            min_eval = float('inf')
            best_column = None
            for column in connect4.possible_drops():
                new_connect4 = self.simulate_drop(connect4, column)
                eval, _ = self.minimax(new_connect4, depth - 1, True)
                if eval < min_eval:
                    min_eval = eval
                    best_column = column
            return min_eval, best_column

    def evaluate(self, connect4):
        if self.my_token == 'o':
            opponent_token = 'x'
        else:
            opponent_token = 'o'

        if connect4.game_over:
            if connect4.wins == self.my_token:
                return 1
            elif connect4.wins == opponent_token:
                return -1
            else:
                return 0

        score = 0


        for four in connect4.iter_fours():
            if four.count(self.my_token) == 3 and four.count('_') == 1:
                score += 100  # Blisko wygranej
            elif four.count(opponent_token) == 3 and four.count('_') == 1:
                score -= 100  # Blisko przegranej

        center_col = connect4.center_column()
        my_center_tokens = center_col.count(self.my_token)
        opponent_center_tokens = center_col.count(opponent_token)
        score += (my_center_tokens - opponent_center_tokens) * 10

        max_possible_score = 100 * 4 + 10 * 3
        min_possible_score = -100 * 4 - 10 * 3

        normalized_score = (score - min_possible_score) / (max_possible_score - min_possible_score) * 2 - 1

        if normalized_score > 1:
            return 0.999
        elif normalized_score < -1:
            return -0.999

        return normalized_score


    def score_line(self, line, player):
        # Określenie oceny dla linii
        score = 0
        opp_player = 1 if player == 2 else 2

        # Sprawdź czy jest możliwa czwórka
        if line.count(player) == 4:
            score += 100
        elif line.count(player) == 3 and line.count(0) == 1:
            score += 5
        elif line.count(player) == 2 and line.count(0) == 2:
            score += 2

        # Sprawdź czy przeciwnik ma możliwą czwórkę
        if line.count(opp_player) == 3 and line.count(0) == 1:
            score -= 4

        return score

    def simulate_drop(self, connect4, column):
        new_connect4 = Connect4(connect4.width, connect4.height)
        new_connect4.board = [row[:] for row in connect4.board]
        new_connect4.who_moves = connect4.who_moves
        new_connect4.game_over = connect4.game_over
        new_connect4.wins = connect4.wins

        new_connect4.drop_token(column)
        return new_connect4

    def get_heuristic_score(self):
        return self.last_heuristic_score