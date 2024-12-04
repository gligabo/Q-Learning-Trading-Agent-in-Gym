# Q-Learning Trading Agent

This project implements a trading agent using Q-learning to learn how to take actions (Short, Hold, Long) based on technical analysis indicators to maximize portfolio returns. The environment is based on OpenAI Gym, allowing for the simulation of various market scenarios to train the agent.

## Project Structure

The project consists of three main scripts:

1. **`data_preprocessing.py`**: Contains the implementation for loading, cleaning, and preparing the market data for training, including feature extraction for technical analysis indicators.
2. **`technical_analysis.py`**: Defines the functions for calculating technical analysis indicators (such as moving averages, RSI, MACD) that the agent will use to make trading decisions.
3. **`trading_qlearning_ofc.py`**: The main script for training the agent using Q-learning, simulating the trading process, and adjusting the agent's policy over time.

## How It Works

### Agent's Actions

The agent can take three possible actions at each time step:

- **Short**: Sell the asset.
- **Hold**: Keep the current position.
- **Long**: Buy the asset.

These actions can be repeated, meaning the agent can decide to "stay" in the same action. For example, if the agent is already in a **Long** position, it may choose to stay in **Long** (keep the position) rather than changing to **Short** or **Hold**. The reward depends on the profitability of the action taken, and the goal is to maximize the total return over time.

### Rewards

The rewards are calculated based on the change in the asset's value after each action, taking into account the financial return of the chosen action. For example, if the agent decides to hold a **Long** position and the asset's value increases, it is rewarded with a positive value. If the asset decreases in value, the reward will be negative. This feedback allows the agent to learn the best trading strategies.

### Technical Indicators

The agent uses various technical indicators to make decisions. Common examples include moving averages, RSI (Relative Strength Index), and MACD (Moving Average Convergence Divergence), which help predict market movements.

---

## Scripts

### 1. `q_learning_agent.py`

This script contains the implementation of the Q-learning agent, which is responsible for making decisions in the trading environment.

```python
import numpy as np
import random

class QLearningAgent:
    def __init__(self, actions, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = {}

    def get_state_key(self, state):
        return tuple(state)

    def update_q_table(self, state, action, reward, next_state):
        current_q = self.q_table.get(self.get_state_key(state), {}).get(action, 0)
        max_future_q = max(self.q_table.get(self.get_state_key(next_state), {}).values(), default=0)
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_future_q - current_q)
        
        if self.get_state_key(state) not in self.q_table:
            self.q_table[self.get_state_key(state)] = {}
        
        self.q_table[self.get_state_key(state)][action] = new_q

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        else:
            state_q_values = self.q_table.get(self.get_state_key(state), {})
            if state_q_values:
                return max(state_q_values, key=state_q_values.get)
            else:
                return random.choice(self.actions)
