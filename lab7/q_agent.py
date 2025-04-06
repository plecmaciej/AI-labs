import numpy as np
from rl_base import Agent, Action, State
import os


class QAgent(Agent):

    def __init__(self, n_states, n_actions,
                 name='QAgent', initial_q_value=0.0, q_table=None):
        super().__init__(name)

        # hyperparams
        # TODO ustaw te parametry na sensowne wartości
        self.lr = 0.1             # learning rate
        self.gamma = 0.99         # discount factor
        self.epsilon = 1.0        # epsilon (probability of random action)
        self.eps_decrement = 0.001 # decrement value for epsilon
        self.eps_min = 0.01       # minimum epsilon value
        #self.lr = 0                # współczynnik uczenia (learning rate)
        #self.gamma = 0             # współczynnik dyskontowania
        #self.epsilon = 0           # epsilon (p-wo akcji losowej)
        #self.eps_decrement = 0     # wartość, o którą zmniejsza się epsilon po każdym kroku
        #self.eps_min = 0           # końcowa wartość epsilon, poniżej którego już nie jest zmniejszane

        self.action_space = [i for i in range(n_actions)]
        self.n_states = n_states
        self.q_table = q_table if q_table is not None else self.init_q_table(initial_q_value)

    def init_q_table(self, initial_q_value=0.):
        # TODO - utwórz tablicę wartości Q o rozmiarze [n_states, n_actions] wypełnioną początkowo wartościami initial_q_value
        q_table = np.full((self.n_states, len(self.action_space)), initial_q_value)
        #q_table = None
        return q_table

    def update_action_policy(self) -> None:
        # TODO - zaktualizuj wartość epsilon
        self.epsilon = max(self.eps_min, self.epsilon - self.eps_decrement)
        #self.epsilon = self.epsilon

    def choose_action(self, state: State) -> Action:

        assert 0 <= state < self.n_states, \
            f"Bad state_idx. Has to be int between 0 and {self.n_states}"

        # TODO - zaimplementuj strategię eps-zachłanną

        if np.random.rand() < self.epsilon:
            # Exploration: choose random action
            action = np.random.choice(self.action_space)
        else:
            # Exploitation: choose the best action based on current Q-values
            action = np.argmax(self.q_table[state])
        return Action(action)

    def learn(self, state: State, action: Action, reward: float, new_state: State, done: bool) -> None:
        # TODO - zaktualizuj q_table
        best_next_action = np.max(self.q_table[new_state])
        td_target = reward + self.gamma * best_next_action
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.lr * td_error

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.q_table)

    def load(self, path):
        self.q_table = np.load(path)

    def get_instruction_string(self):
        return [f"Linearly decreasing eps-greedy: eps={self.epsilon:0.4f}"]
