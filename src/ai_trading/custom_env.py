
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TradingEnv(gym.Env):
    def __init__(self, data, initial_balance=10000):
        super(TradingEnv, self).__init__()

        # Discrete actions: 0 (Open Long), 1 (Close Position), 2 (Open Short), 3 (Hold Position), 4 (Partial Close), 5 (Await Entry)
        self.action_space = spaces.Tuple((
            spaces.Discrete(6),  # 5 discrete actions
            spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)  # Continuous space for partial close (only used when action == 4)
        ))

        # Define the observation space (e.g., features, position state, time in position, profit, position size)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

        # Price data (for simplicity)
        self.data = data
        self.current_step = 0
        self.done = False
        
        # Trading related attributes
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position_state = 0  # 1 for long, -1 for short, 0 for no position
        self.time_in_position = 0
        self.position_open_price = None
        self.position_size = 0  # New variable to track position size (fraction of the full position)
        self.pnl = 0

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position_state = 0
        self.position_size = 0
        self.time_in_position = 0
        self.position_open_price = None
        self.pnl = 0
        self.done = False
        return self._next_observation()

    def _next_observation(self):
        step_features = self.data[self.current_step]
        return np.array([step_features, self.position_state, self.time_in_position, self.pnl, self.position_size])

    def _take_action(self, action):
        discrete_action = action[0]  # Discrete action
        partial_close_percentage = action[1][0]  # Continuous action (partial close percentage)

        feature_set = self.data[self.current_step]

        if discrete_action == 0:  # Open Long
            
            if self.position_state != 0:
                self._close_position(feature_set.price)
                
            self.position_state = 1
            self.position_open_price = feature_set.price
            self.position_open_price = feature_set.price
            self.position_size = 1  # Full position size
            self.time_in_position = 0
            self.pnl = 0

        
        elif discrete_action == 1:  # Close Position
            if self.position_state != 0:
                self._close_position(feature_set.price)
        
        elif discrete_action == 2:  # Open Short
            
            if self.position_state != 0:
                # Trade Reversal
                self._close_position(feature_set.price)
                
            self.position_state = -1
            self.position_open_price = feature_set.price
            self.position_size = 1  # Full position size
            self.time_in_position = 0
            self.pnl = 0
                
                
        
        elif discrete_action == 3:  # Hold Position
            if self.position_state != 0:
                self.time_in_position += 1
                self._update_pnl(feature_set.price)

        elif discrete_action == 4:  # Partial Close
            if self.position_state != 0 and 0 < partial_close_percentage <= 1:
                self._partial_close_position(feature_set.price, partial_close_percentage)
        
    def _partial_close_position(self, current_price, percentage):
        self._update_pnl(current_price)
        self.balance += self.pnl * percentage  # Update balance based on partial profit
        self.position_size -= percentage  # Reduce position size by the closed amount
        if self.position_size <= 0:  # Fully closed
            self.position_state = 0
            self.position_size = 0
            self.time_in_position = 0
            self.position_open_price = None
            self.pnl = 0
            
    def _update_pnl(self, current_price):
        if self.position_state == 1:  # Long position
            self.pnl = ((current_price - self.position_open_price) / self.position_open_price) * self.position_size
        elif self.position_state == -1:  # Short position
            self.pnl = ((self.position_open_price - current_price) / self.position_open_price) * self.position_size
    
    def _close_position(self, current_price):
        self._update_pnl(current_price)
        self.balance += self.pnl  # Update balance based on profit/loss from the position
        self.position_state = 0
        self.position_size = 0
        self.time_in_position = 0
        self.position_open_price = None
        self.pnl = 0

    def step(self, action):
        self._take_action(action)

        # Move to the next step
        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True

        # Reward based on profit or loss in the position
        reward = 0
        if self.position_state != 0:
            if self.pnl > 0:
                reward = self.in_money_factor * self.pnl * self.time_in_position
            elif self.pnl < 0:
                reward = self.out_of_money_factor * abs(self.pnl) * self.time_in_position

        # Get the next observation
        observation = self._next_observation()

        return observation, reward, self.done, {}

    def render(self, mode='human'):
        print(f'Step: {self.current_step}')
        print(f'Price: {self.data[self.current_step]}')
        print(f'Position: {"Long" if self.position_state == 1 else "Short" if self.position_state == -1 else "None"}')
        print(f'Time in Position: {self.time_in_position}')
        print(f'Profit: {self.pnl}')
        print(f'Balance: {self.balance}')
        print(f'Position Size: {self.position_size}')

# Example price data for the environment
price_data = np.random.uniform(100, 200, size=1000)

# Create the environment
env = TradingEnv(price_data)

# Example of running a random agent
observation = env.reset()
for _ in range(100):
    action = env.action_space.sample()  # Random action
    observation, reward, done, info = env.step(action)
    env.render()
    if done:
        break
