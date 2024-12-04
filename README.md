# Q-Learning Trading Agent

This project implements a trading agent using Q-learning to learn how to take actions (Short, Hold, Long) based on technical analysis indicators to maximize portfolio returns. The environment is based on OpenAI Gym, allowing for the simulation of various market scenarios to train the agent.

## Project Structure

The project consists of three main scripts:

1. **`data_preprocessing.py`**: Contains the implementation for loading and cleaning the data
2. **`technical_analysis.py`**: Defines the functions for calculating technical analysis indicators (such as moving averages, RSI, MACD) that the agent will use to make trading decisions. Also discretize the states and uses rolling windows to avoid the influence of currency value flutuations over time.
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

The agent uses various technical indicators to make decisions. Common examples include moving averages, RSI (Relative Strength Index), MACD (Moving Average Convergence Divergence), and Bollinger Bands, which help predict market trends and potential price movements. Bollinger Bands, in particular, are useful for identifying overbought or oversold conditions by measuring the volatility of an asset relative to its moving average.

### Results visualization

The results after training can be visualized using the render method from the gym-trading-env library. This method displays the actions taken by the agent, the portfolio value over time, and the corresponding rewards, allowing you to track the agent's performance and decision-making process throughout the trading episodes.
