# Markov Decision Process Assignment

## Introduction

This assignment involves implementing and analyzing a Markov Decision Process (MDP) on a 3x3 grid environment. The goal is to find the optimal policy and value function for different reward values using value iteration and policy iteration algorithms.

## Grid Environment
The GridWorld environment is a 3x3 grid where each cell has a reward value. The agent can take one of four actions: Up (U), Down (D), Left (L), or Right (R). The agent's movement is probabilistic, with an 80% chance of moving in the intended direction and a 10% chance of moving to the left or right of the intended direction.

### Action Transition Probabilities
We define the action transition probabilities as follows:
```python
self.transition_probs = {
            'U': [(-1, 0, 0.8), (0, -1, 0.1), (0, 1, 0.1)],     # Main, left, right
            'D': [(1, 0, 0.8), (0, 1, 0.1), (0, -1, 0.1)],      # Main, left, right
            'L': [(0, -1, 0.8), (1, 0, 0.1), (-1, 0, 0.1)],     # Main, Up, Down
            'R': [(0, 1, 0.8), (1, 0, 0.1), (-1, 0, 0.1)]       # Main, Up, Down
        }
```
Where the main transition probability is 80%, and the left and right transition probabilities are 10% each.

### Reward Matrix

The reward matrix for the GridWorld is defined as follows:

```
[
    [r, -1, 10],
    [-1, -1, -1],
    [-1, -1, -1]
]
```

Where r and 10 are terminal states, and therefore, we do not update their values during value iteration.

`r is a variable reward value that we will test with different values (100, 3, 0, -3).`

## Implementation

The implementation is provided in the [main.py](main.py) file. The key components of the implementation are:

1. **GridWorld Class**: The main class with methods to initialize the environment, calculate the transition probabilities, and perform value iteration and policy iteration.

### Helper Methods
is_valid_state: Checks if the given state is valid or not. i.e., will the next state hit a wall or go out of bounds.
get_next_state: Returns the next state based on the current state and action.

### Expected Value Calculation
The expected value calculation is done using the following formula:
```python
def calculate_expected_value(self, state, action, value_function):
        next_states = self.get_next_state(state, action)
        expected_value = 0.0    
        for (nx, ny), prob in next_states:
            expected_value += prob * value_function[nx, ny]
        return expected_value
```
Which is the second part to the Bellman equation for the value function:
$$V_{k+1}(s) = R_s^a + \gamma \sum_{s'} P(s'|s) V(s')$$
- **$\gamma$** is the discount factor  is set to `0.99` as per assignment instructions
- **$R_s^a$** is the reward matrix which is added in our implementation 

### Value Iteration

The value iteration pseudocode is as follows:
```pseudo
Function value_iteration():
    Initialize value_function as a zero matrix of size (grid_size, grid_size)
    Set value_function[0, 2] to reward_matrix[0, 2]  # Terminal state
    Set value_function[0, 0] to reward_matrix[0, 0]  # Terminal state
    Initialize policy as a zero matrix of size (grid_size, grid_size) with string type

    Loop indefinitely:
        Set delta to 0
        For each state (x, y) in the grid:
            If (x, y) is a terminal state:
                Continue to the next state
            Set v to value_function[x, y]
            Set max_value to negative infinity
            Set best_action to None

            For each action in actions:
                Calculate value as reward_matrix[x, y] + discount_factor * expected_value
                If value is greater than max_value:
                    Set max_value to value
                    Set best_action to action

            Set value_function[x, y] to max_value
            Set policy[x, y] to best_action
            Update delta to the maximum of delta and the absolute difference between v and max_value

        If delta is less than a small threshold (1e-6):
            Break the loop

    Return value_function and policy
```

### Policy Iteration
Policy iteration is split into two parts: policy evaluation and policy improvement. 

#### Policy Evaluation
The policy evaluation pseudocode is as follows:
```pseudo
Function policy_evaluation(policy):
    initialize value_function as a zero matrix of grid_size and add terminal state rewards
    Loop indefinitely:
        Set delta to 0
        For each state (x, y) in the grid:
            If (x, y) is a terminal state:
                Continue to the next state
            Set v to value_function[x, y]
            Set action to policy[x, y]
            Set value_function[x, y] to calculate_expected_value(x, y, action, value_function)
            Update delta to the maximum of delta and the absolute difference between v and value_function[x, y]

        If delta is less than a small threshold (1e-6):
            Break the loop

    Return value_function
```

#### Policy Improvement
The policy improvement pseudocode is as follows:
```pseudo
Function policy_improvement(value_function):
    Initialize policy as a zero matrix of size (grid_size, grid_size) for each state (x, y) in the grid:
        If (x, y) is a terminal state:
            Continue to the next state
        Set max_value to negative infinity
        Set best_action to None

        For each action in actions:
            Calculate value as reward_matrix[x, y] + discount_factor * expected_value
            If value is greater than max_value:
                Set max_value to value
                Set best_action to action

        Set policy[x, y] to best_action

    Return policy
```

### Policy Iteration
The policy iteration combines the policy evaluation and policy improvement steps. The pseudocode is as follows:
```pseudo
Function policy_iteration():
    Initialize policy as a random policy
    Loop indefinitely:
        Set value_function to policy_evaluation(policy)
        Set new_policy to policy_improvement(value_function)

        If new_policy is equal to policy:
            Break the loop

        Set policy to new_policy

    Return value_function and policy
```

### Testing

The implementation tests the GridWorld environment with different reward values `r = 100, 3, 0, -3` using both value iteration and policy iteration algorithms. The results are printed to the console. 

### Results

#### **1. \(r = 100\)**

##### 1.1 Value Iteration
```
Values:
[[100.          97.20352595  10.        ]
 [ 97.20352595  94.75128164  88.20426077]
 [ 94.48183415  92.35292955  89.76219978]]

Policy:
[['' 'L' '']
 ['U' 'L' 'D']
 ['U' 'L' 'L']]
```

##### 1.2 Policy Iteration
```
Values:
[[100.          97.20352595  10.        ]
 [ 97.20352595  94.75128164  88.20426073]
 [ 94.48183414  92.35292954  89.76219977]]

Policy:
[['' 'L' '']
 ['U' 'L' 'D']
 ['U' 'L' 'L']]
```

---

#### **2. \(r = 3\)**

##### 2.1 Value Iteration
```
Values:
[[ 3.          8.46193925 10.        ]
 [ 5.38356479  7.11320486  8.46193927]
 [ 4.57481543  5.79411117  6.96500877]]

Policy:
[['' 'R' '']
 ['R' 'R' 'U']
 ['R' 'R' 'U']]
```

##### 2.2 Policy Iteration
```
Values:
[[ 3.          8.46193926 10.        ]
 [ 5.38356485  7.11320487  8.46193927]
 [ 4.57481553  5.79411119  6.96500877]]

Policy:
[['' 'R' '']
 ['R' 'R' 'U']
 ['R' 'R' 'U']]
```

---

#### **3. \(r = 0\)**

##### 3.1 Value Iteration
```
Values:
[[ 0.          8.46193927 10.        ]
 [ 5.08329869  7.11320489  8.46193927]
 [ 4.54182306  5.79411123  6.96500878]]

Policy:
[['' 'R' '']
 ['R' 'R' 'U']
 ['R' 'R' 'U']]
```

##### 3.2 Policy Iteration
```
Values:
[[ 0.          8.46193926 10.        ]
 [ 5.08329858  7.11320487  8.46193927]
 [ 4.5418229   5.79411119  6.96500877]]

Policy:
[['' 'R' '']
 ['R' 'R' 'U']
 ['R' 'R' 'U']]
```

---

#### **4. \(r = -3\)**

##### 4.1 Value Iteration
```
Values:
[[-3.          8.46193927 10.        ]
 [ 4.78303242  7.11320489  8.46193927]
 [ 4.50883042  5.79411123  6.96500878]]

Policy:
[['' 'R' '']
 ['R' 'R' 'U']
 ['R' 'R' 'U']]
```

##### 4.2 Policy Iteration
```
Values:
[[-3.          8.46193926 10.        ]
 [ 4.78303231  7.11320487  8.46193927]
 [ 4.50883026  5.79411119  6.96500877]]

Policy:
[['' 'R' '']
 ['R' 'R' 'U']
 ['R' 'R' 'U']]
```

---

## Key Observations

1. **Consistency Between Algorithms:**  
   - The values and policies generated by Value Iteration and Policy Iteration are nearly identical across all reward settings (\(r\)).
   - Minor differences in values are due to numerical precision.

2. **Impact of \(r\) on Policy:**  
   - For \(r = 100\), the agent prefers moving towards the high-reward state aggressively (indicated by the leftward and upward policies).  
   - For lower or negative \(r\), the agent prioritizes moving towards the higher-valued states while avoiding negative rewards.
