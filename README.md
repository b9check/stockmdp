# Stock Action Optimization with Markov Decision Processes

## Overview
This project uses a Markov Decision Process (MDP) to determine optimal buy, sell, or hold strategies for stocks. The approach leverages historical stock data, calculates Relative Strength Index (RSI), and applies value iteration to derive policies that maximize expected returns.

## Features
- Calculates RSI-based stock momentum states: **Oversold**, **Neutral**, **Overbought**.
- Classifies actions as **Buy**, **Sell**, or **Hold** based on deviations from expected prices.
- Constructs transition (T) and reward (R) matrices for a stock.
- Applies value iteration to determine optimal policies.
- Compares the optimal policy against 27 naive policies.
- Includes functionality for meta-analysis across multiple stocks or single stock evaluation.

### Transition and Reward Matrices
Using labeled weekly data, the framework builds:
- **Transition Matrix (T):** Captures the probability of transitioning between states given a specific action. For example, the probability of transitioning from "Oversold" to "Neutral" after a "Buy" action is calculated as:
  
T(s, a, s') = (Number of transitions from state s to s' after action a) / (Total number of times action a was taken in state s)

- **Reward Matrix (R):** Records the expected percentage price change for each state-action pair. For example, the reward for taking a "Buy" action in the "Oversold" state is calculated as the average observed percentage change in price for such occurrences.

### Value Iteration
The framework applies **value iteration** to compute the optimal policy. This algorithm maximizes the expected long-term reward for each state:

a*(s) = argmax_a [R(s, a) + γ * Σ(T(s, a, s') * U(s'))]

Where:
- `a*(s)`: Optimal action for state `s`.
- `γ`: Discount factor (set to 0.9).
- `U(s')`: Utility of transitioning to state `s'`.

### Policy Evaluation
Each stock's optimal policy is compared against all 27 naive policies, such as always "Buy," "Sell," or "Hold." Total returns for each policy are calculated over the testing period.

## Functions

### `calculate_rsi(daily_data, length=14)`
- Computes the Relative Strength Index (RSI) for the given stock's daily closing prices.
- Returns weekly aggregated RSI values.

### `TR_matrices_from_data(ticker, start_date, end_date)`
- Downloads historical stock price data from Yahoo Finance.
- Computes the transition (`T`) and reward (`R`) matrices based on labeled states and actions.
- Returns `T`, `R`, state mappings, and state occurrence counts.

### `value_iteration(T, R, threshold=0.01, gamma=0.9)`
- Performs value iteration on the `T` and `R` matrices to derive the optimal policy.
- Returns the policy as a list of optimal actions for each state.

### `evaluate_policy(policy, ticker, test_start_date, test_end_date)`
- Calculates the total expected return of a given policy during the testing period.
- Returns the total return.

### `evaluate_all_policies(ticker, test_start_date, test_end_date)`
- Evaluates all 27 naive policies for a given stock over the testing period.
- Returns a sorted list of policies with their total returns.

### `plot_policy_histogram(sorted_results, optimal_policy, ticker)`
- Visualizes the performance of all policies as a histogram.
- Highlights the optimal policy in a distinct color.
- Saves the plot as an image.

### `policy_meta_analysis()`
- Analyzes the effectiveness of the optimal policy versus naive policies across the 20 largest companies by market cap.
- Aggregates and normalizes returns for each policy.
- Generates a comparative histogram and returns aggregated results.

## Usage

### Meta-Analysis
Run the meta-analysis across multiple stocks:

python final_project.py meta_analysis

### Single-Stock-Analysis
Run the analysis across a single stock:

python final_project.py stock_analysis

