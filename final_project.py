import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import linregress
import matplotlib.pyplot as plt
import itertools
import sys


def calculate_rsi(daily_data, length=14):
    # Use formula to calculate average RSI for a given week
    delta = daily_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / loss
    daily_data['RSI'] = 100 - (100 / (1 + rs))
    weekly_rsi = daily_data['RSI'].resample('W').mean()

    return weekly_rsi


def TR_matrices_from_data(ticker, start_date, end_date):
    # Import raw data from Yahoo Finance
    data = yf.download(ticker, start=start_date, end=end_date, interval="1wk")
    data.sort_index(inplace=True)
    data['Pct_change'] = data['Close'].pct_change()

    # Create RSI and set thresholds for overbought/oversold
    daily_data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
    weekly_rsi = calculate_rsi(daily_data).reindex(data.index, method='nearest')
    data['RSI'] = weekly_rsi
    oversold_threshold = data['RSI'].mean() - data['RSI'].std()
    overbought_threshold = data['RSI'].mean() + data['RSI'].std()

    # Calculate slope, variance, and expected next value
    data['Slope'] = np.nan
    data['STD'] = np.nan
    data['Expected_close'] = np.nan
    rolling_window = 12

    for i in range(rolling_window + 1, len(data)):
        y = data['Close'].iloc[i - 1 - rolling_window:i - 1].values.flatten()
        x = np.arange(len(y))
        slope, intercept, _, _, _ = linregress(x, y)
        variance = np.var(y)

        expected_close = data['Open'].iloc[i].item() + slope
        data.loc[data.index[i], 'Slope'] = slope
        data.loc[data.index[i], 'STD'] = variance ** 0.5
        data.loc[data.index[i], 'Expected_close'] = expected_close

    data.dropna(axis=0, how='any', subset=None, inplace=True)

    # 'new_data' matrix stores state, action, and reward (price change) each time step
    new_data = pd.DataFrame(index=data.index)

    # Calculate if each day was a 'buy', 'sell', 'neutral', or 'none' for each day
    for i in range(len(data)):
        if data['Close'].iloc[i].item() > (data['Expected_close'].iloc[i].item() + 0.5 * data['STD'].iloc[i].item()):
            new_data.loc[data.index[i], 'Action'] = 'buy'
        elif data['Close'].iloc[i].item() < (data['Expected_close'].iloc[i].item() - 0.5 * data['STD'].iloc[i].item()):
            new_data.loc[data.index[i], 'Action'] = 'sell'
        elif abs(data['Close'].iloc[i].item() - (data['Expected_close'].iloc[i].item())) < 0.25 * data['STD'].iloc[i].item():
            new_data.loc[data.index[i], 'Action'] = 'hold'
        else:
            new_data.loc[data.index[i], 'Action'] = 'none'

    new_data['State'] = np.where(data['RSI'] > overbought_threshold, 'overbought', 
                                 np.where(data['RSI'] < oversold_threshold, 'oversold', 'neutral'))
    new_data['Pct_change'] = data['Pct_change']
    new_data = new_data[new_data['Action'] != 'none']

    # Define possible states and actions
    states = ['oversold', 'neutral', 'overbought']
    actions = ['buy', 'sell', 'hold']

    # Initialize matrices for transitions (T) and rewards (R)
    num_states = len(states)
    num_actions = len(actions)
    T = np.zeros((num_states, num_actions, num_states))
    R = np.zeros((num_states, num_actions))

    state_mapping = {state: i for i, state in enumerate(states)}
    action_mapping = {action: i for i, action in enumerate(actions)}

    # Loop through new_data and create counts for transitions and rewards
    for i in range(len(new_data) - 1):
        state = state_mapping[new_data.iloc[i]['State']]
        action = action_mapping[new_data.iloc[i]['Action']]
        next_state = state_mapping[new_data.iloc[i + 1]['State']]
        pct_change = new_data.iloc[i + 1]['Pct_change']

        T[state, action, next_state] += 1
        R[state, action] += pct_change

    # Store current matrices as counts for evaluation
    state_counts = np.sum(T, axis=(1, 2))

    # Normalize T and R matrices
    for s in range(num_states):
        for a in range(num_actions):
            total = np.sum(T[s, a, :])
            if total > 0:
                T[s, a, :] /= total  
                R[s, a] /= total  

    return T, R, state_mapping, state_counts


def value_iteration(T, R, threshold: float, gamma: float):
    num_states = R.shape[0]
    num_actions = R.shape[1]

    # Value iteration to compute utility of each state
    U = np.zeros(num_states)
    delta = float('inf')
    while delta > threshold:
        delta = 0
        for s in range(num_states):
            v = U[s]
            U[s] = max(R[s, a] + gamma * sum(T[s, a, sp] * U[sp] for sp in range(num_states)) for a in range(num_actions))
            delta = max(delta, abs(v - U[s]))
    
    # Choose action that maximizes expected reward for each state
    policy = np.zeros(num_states, dtype=int)
    for s in range(num_states):
        policy[s] = np.argmax([R[s, a] + gamma * sum(T[s, a, sp] * U[sp] for sp in range(num_states)) for a in range(num_actions)])

    # Convert policy to words
    action_mapping_inv = {i: action for i, action in enumerate(['buy', 'sell', 'hold'])}
    policy_in_words = [action_mapping_inv[action] for action in policy]

    return policy_in_words



def evaluate_policy(policy, ticker, test_start_date, test_end_date):
    _, R, state_mapping, state_counts = TR_matrices_from_data(ticker, test_start_date, test_end_date)    
    action_mapping = {'buy': 0, 'sell': 1, 'hold': 2}
    
    # Compute total reward for the policy: [R(s, pi(s)) * total_state_appearances] summed over states
    total_reward = 0
    for state_name, action_name in zip(state_mapping.keys(), policy):
        state_index = state_mapping[state_name]
        action_index = action_mapping[action_name]
        total_reward += R[state_index, action_index] * state_counts[state_index]

    return total_reward



def evaluate_all_policies(ticker, test_start_date, test_end_date):
    actions = ['buy', 'sell', 'hold']
    all_policies = list(itertools.product(actions, repeat=3)) 
    results = []

    # Loop through all policies and evaluate them on unseen test data
    for policy in all_policies:
        policy = list(policy)
        total_return = evaluate_policy(policy, ticker, test_start_date, test_end_date)
        results.append((policy, 100*total_return))

    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

    return sorted_results



def plot_policy_histogram(sorted_results, optimal_policy, ticker):
    # Extract policies and their returns
    policies = [result[0] for result in sorted_results]
    returns = [result[1] for result in sorted_results]

    # Identify the index of the optimal policy
    optimal_index = policies.index(optimal_policy)

    # Create bar colors: gray for others, a distinct color for the optimal policy
    colors = ['gray'] * len(policies)
    colors[optimal_index] = 'blue'

    # Plot the histogram
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(policies)), returns, color=colors)

    # Highlight the optimal policy
    plt.xticks(range(len(policies)), [str(policy) for policy in policies], rotation=90, fontsize=8)
    plt.ylabel('Total Return (%)')
    plt.title(f'Policy Performance Histogram for {ticker} (Optimal Highlighted)')
    plt.tight_layout()
    plt.savefig(f"{ticker}_policy_histogram.png", dpi=300)
    plt.show()


def policy_meta_analysis():
    train_start_date = "2000-01-01"
    train_end_date = "2020-01-01"
    test_start_date = "2023-01-01"
    test_end_date = "2024-12-01"
    largest_stock_tickers = ["AAPL", "NVDA", "MSFT", "AMZN", "GOOG", "META", "TSLA", "BRK.B", "UNH", "XOM", "JNJ", "V", "PG", "MA", "HD", "CVX", "KO", "PEP", "ORCL", "JPM"]
    
    # Initialize storage for results
    overall_policy_results = {tuple(policy): [] for policy in itertools.product(['buy', 'sell', 'hold'], repeat=3)}
    overall_policy_results["Optimal"] = []


    for ticker in largest_stock_tickers:
        try:
            # Train: Calculate T, R matrices and determine the optimal policy
            T, R, state_mapping, state_counts = TR_matrices_from_data(ticker, train_start_date, train_end_date)
            optimal_policy = value_iteration(T, R, threshold=0.01, gamma=0.9)

            # Test: Evaluate all policies on test data
            sorted_results = evaluate_all_policies(ticker, test_start_date, test_end_date)

            # Normalize returns
            max_return = max(ret for _, ret in sorted_results)
            for policy, total_return in sorted_results:
                normalized_return = total_return / max_return
                overall_policy_results[tuple(policy)].append(normalized_return)

            # Normalize and store the optimal policy's return
            optimal_policy_return = next(ret for policy, ret in sorted_results if policy == optimal_policy)
            normalized_optimal_return = optimal_policy_return / max_return
            overall_policy_results["Optimal"].append(normalized_optimal_return)
            print(f'{ticker}: Optimal policy {optimal_policy} results in {normalized_optimal_return}')
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    # Average out and sort returns for each policy
    averaged_policy_results = {policy: sum(returns) / len(returns) if returns else 0
                           for policy, returns in overall_policy_results.items()}
    sorted_policies = sorted(averaged_policy_results.items(), key=lambda x: x[1], reverse=True)
    policies, returns = zip(*sorted_policies)

    # Create the histogram
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(returns)), returns, color='gray')
    optimal_index = policies.index('Optimal')
    bars[optimal_index].set_color('red')

    plt.xticks(range(len(policies)), ['-'.join(policy) if isinstance(policy, tuple) else policy for policy in policies], rotation=90)
    plt.xlabel('Policies')
    plt.ylabel('Average Normalized Return')
    plt.title('Average Normalized Returns for All Policies Across 20 Largest Companies (Optimal Highlighted)')
    plt.tight_layout()
    plt.savefig("policy_histogram.png", dpi=300)
    plt.show()

    return averaged_policy_results




if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("Usage: python final_project.py 'meta_analysis' or 'stock_analysis'")
    goal = sys.argv[1]  # Input CSV file
    
    if goal == "meta_analysis":
        policy_meta_analysis()

    if goal == "stock_analysis":
        # INPUTS: ticker, training dates, and testing dates
        ticker = "NVDA"
        train_start_date = "2000-01-01"
        train_end_date = "2020-01-01"
        test_start_date = "2023-01-01"
        test_end_date = "2024-12-01"
        
        # Get transition and reward matrices from data
        T, R, state_mapping, state_counts = TR_matrices_from_data(ticker, train_start_date, train_end_date)

        # Find the optimal policy using value iteration
        optimal_policy = value_iteration(T, R, threshold=0.01, gamma=0.9)

        # Evaluate policy by comparing to all possible policies
        sorted_returns = evaluate_all_policies(ticker, test_start_date, test_end_date)
        plot_policy_histogram(sorted_returns, optimal_policy, ticker)

        # Fetch the latest available data for the ticker and determine best action
        latest_data = data = yf.download(ticker, period="1mo", interval="1d")
        weekly_rsi = calculate_rsi(latest_data).reindex(data.index, method='nearest')
        latest_data['RSI'] = weekly_rsi

        if latest_data['RSI'].iloc[-1] < latest_data['RSI'].mean() - latest_data['RSI'].std():
            current_state = "oversold"
        elif latest_data['RSI'].iloc[-1] > latest_data['RSI'].mean() + latest_data['RSI'].std():
            current_state = "overbought"
        else:
            current_state = "neutral"

        optimal_action = optimal_policy[state_mapping[current_state]]
        print(f"Based on the latest data, the current state is '{current_state}', and the optimal action is '{optimal_action}'.")