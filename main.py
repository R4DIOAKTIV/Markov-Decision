import numpy as np

class GridWorld:
    def __init__(self, grid_size, reward_matrix, discount_factor=0.99):
        self.grid_size = grid_size
        self.reward_matrix = np.array(reward_matrix)
        self.discount_factor = discount_factor
        self.actions = ['U', 'D', 'L', 'R']
        self.transition_probs = {
            'U': [(-1, 0, 0.8), (0, -1, 0.1), (0, 1, 0.1)],  # Main, left, right
            'D': [(1, 0, 0.8), (0, 1, 0.1), (0, -1, 0.1)],
            'L': [(0, -1, 0.8), (1, 0, 0.1), (-1, 0, 0.1)],
            'R': [(0, 1, 0.8), (1, 0, 0.1), (-1, 0, 0.1)]
        }

    def is_valid_state(self, state):
        x, y = state
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def get_next_state(self, state, action):
        x, y = state
        transitions = self.transition_probs[action]
        next_states = []

        for dx, dy, prob in transitions:
            nx, ny = x + dx, y + dy
            if self.is_valid_state((nx, ny)):
                next_states.append(((nx, ny), prob))
            else:
                next_states.append((state, prob))  # Stay in place on invalid move

        return next_states

    def calculate_expected_value(self, state, action, value_function):
        next_states = self.get_next_state(state, action)
        expected_value = 0.0

        for (nx, ny), prob in next_states:
            expected_value += prob *(self.reward_matrix[nx, ny] + self.discount_factor * value_function[nx, ny])
            # expected_value += prob * value_function[nx, ny]
        return expected_value

    def value_iteration(self):
        value_function = np.zeros((self.grid_size, self.grid_size))
        policy = np.zeros((self.grid_size, self.grid_size), dtype=str)

        while True:
            delta = 0
            for x in range(self.grid_size):
                for y in range(self.grid_size-1,-1,-1):
                    # if x == 0 and y == 2:
                    #     continue
                    v = value_function[x, y]
                    max_value = float('-inf')
                    best_action = None

                    for action in self.actions:
                        # expected_value = self.calculate_expected_value((x, y), action, value_function)
                        # value = self.reward_matrix[x, y] + self.discount_factor * expected_value
                        value = self.calculate_expected_value((x, y), action, value_function)
                        if value > max_value:
                            max_value = value
                            best_action = action

                    value_function[x, y] = max_value
                    policy[x, y] = best_action
                    delta = max(delta, abs(v - max_value))

            if delta < 1e-6:
                break

        return value_function, policy

    def policy_evaluation(self, policy):
        value_function = np.zeros((self.grid_size, self.grid_size))

        while True:
            delta = 0
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    # if x == 0 and y == 2:
                    #     continue
                    v = value_function[x, y]
                    action = policy[x, y]
                    expected_value = self.calculate_expected_value((x, y), action, value_function)
                    value_function[x, y] = self.reward_matrix[x, y] + self.discount_factor * expected_value
                    delta = max(delta, abs(v - value_function[x, y]))

            if delta < 1e-6:
                break
            
        return value_function

    def policy_improvement(self, value_function):
        policy = np.zeros((self.grid_size, self.grid_size), dtype=str)

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                max_value = float('-inf')
                best_action = None

                for action in self.actions:
                    expected_value = self.calculate_expected_value((x, y), action, value_function)
                    value = self.reward_matrix[x, y] + self.discount_factor * expected_value

                    if value > max_value:
                        max_value = value
                        best_action = action

                policy[x, y] = best_action

        return policy

    def policy_iteration(self):
        policy = np.random.choice(self.actions, (self.grid_size, self.grid_size))

        while True:
            value_function = self.policy_evaluation(policy)
            new_policy = self.policy_improvement(value_function)

            if np.array_equal(policy, new_policy):
                break

            policy = new_policy

        return value_function, policy

# Test with different r values
r_values = [100, 3, 0, -3]
for r in r_values:
    reward_matrix = [
        [r, -1, 10],
        [-1, -1, -1],
        [-1, -1, -1]
    ]
    grid_world = GridWorld(3, reward_matrix)

    # Value Iteration
    value_function_vi, policy_vi = grid_world.value_iteration()
    print(f"Value Iteration for r = {r}:")
    print("Values:")
    print(value_function_vi)
    print("Policy:")
    print(policy_vi)
    print("-" * 20)

    # Policy Iteration
    value_function_pi, policy_pi = grid_world.policy_iteration()
    print(f"Policy Iteration for r = {r}:")
    print("Values:")
    print(value_function_pi)
    print("Policy:")
    print(policy_pi)
    print("=" * 20)
