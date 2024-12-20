import numpy as np

class GridWorld:
    def __init__(self, grid_size, reward_matrix, discount_factor=0.99):
        self.grid_size = grid_size
        self.reward_matrix = reward_matrix
        self.discount_factor = discount_factor
        self.actions = ['U', 'D', 'L', 'R']
        self.transition_probs = {
            'U': ((0, 1), 0.8),  # (dx, dy), probability
            'D': ((0, -1), 0.8),
            'L': [(-1, 0), 0.8],
            'R': [(1, 0), 0.8]
        }

    def is_valid_state(self, state):
        # Check if the state is within the grid boundaries
        x, y = state
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

    def get_next_state(self, state, action):
        # Get the next state based on the current state and action
        x, y = state
        dx, dy = self.transition_probs[action][0]
        next_x, next_y = x + dx, y + dy
        if not self.is_valid_state((next_x, next_y)):
            return state  # If next state is invalid, stay in the current state
        return next_x, next_y

    def value_iteration(self):
        # Perform value iteration to find the optimal value function and policy
        value_function = np.zeros((self.grid_size, self.grid_size))
        policy = np.zeros((self.grid_size, self.grid_size), dtype=str)
        while True:
            delta = 0
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    v = value_function[i, j]
                    max_value = float('-inf')
                    max_action = None
                    for action in self.actions:
                        next_state = self.get_next_state((i, j), action)
                        next_value = value_function[next_state[0], next_state[1]]
                        value = self.reward_matrix[i][j] + self.discount_factor * next_value
                        if value > max_value:
                            max_value = value
                            max_action = action
                    value_function[i, j] = max_value
                    policy[i, j] = max_action
                    delta = max(delta, abs(v - value_function[i, j]))
            if delta < 1e-6:
                break
        return value_function, policy

    def policy_evaluation(self, policy):
        # Evaluate the given policy to find the value function
        value_function = np.zeros((self.grid_size, self.grid_size))
        while True:
            delta = 0
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    v = value_function[i, j]
                    action = policy[i, j]
                    next_state = self.get_next_state((i, j), action)
                    next_value = value_function[next_state[0], next_state[1]]
                    value_function[i, j] = self.reward_matrix[i][j] + self.discount_factor * next_value
                    delta = max(delta, abs(v - value_function[i, j]))
            if delta < 1e-6:
                break
        return value_function

    def policy_improvement(self, value_function):
        # Improve the policy based on the given value function
        policy = np.zeros((self.grid_size, self.grid_size), dtype=str)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                max_value = float('-inf')
                best_action = None
                for action in self.actions:
                    next_state = self.get_next_state((i, j), action)
                    next_value = value_function[next_state[0], next_state[1]]
                    value = self.reward_matrix[i][j] + self.discount_factor * next_value
                    if value > max_value:
                        max_value = value
                        best_action = action
                policy[i, j] = best_action
        return policy

    def policy_iteration(self):
        # Perform policy iteration to find the optimal value function and policy
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