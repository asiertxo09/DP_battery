import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
np.random.seed(10)

def battery_optimization(prices, capacity, max_rate, grid_limit):
    n = len(prices)
    min_soc = int(0.4 * capacity)  # Minimum SoC at the end of the day
    max_soc = int(0.6 * capacity)  # Maximum SoC at the end of the day
    rates = np.linspace(-max_rate, max_rate, 50)  # Possible rates: Discharge, Do Nothing, Charge
    memo = {}
    actions = {}
    profit_tracking = {}

    def dp(t, charge):
        # Base case: If we've reached the end of time steps
        if t == n:
            return 0 if min_soc <= charge <= max_soc else float('-inf')

        # Check memoization table for already computed state
        if (t, charge) in memo:
            return memo[(t, charge)]

        # Initialize best variables
        best_action = float('-inf')
        best_rate = 0
        best_profit = 0

        # Iterate over all possible rates
        for rate in range(-max_rate, max_rate,5):
            new_charge = charge + rate

            # Ensure new_charge is within valid bounds
            if 0 <= new_charge <= capacity and rate <= grid_limit:
                # Calculate the immediate profit based on the rate
                if rate > 0 and prices[t] < 0:  # Charging at negative prices
                    immediate_profit = abs(rate) * abs(prices[t])
                elif rate < 0 and prices[t] > 0:  # Discharging at positive prices
                    immediate_profit = abs(rate) * prices[t]
                else:  # Do nothing or invalid rate
                    immediate_profit = -(rate * prices[t])

                # Recursive call to calculate the total profit
                total_profit = immediate_profit + dp(t + 1, new_charge)

                # Update the best action if this rate is optimal
                if total_profit > best_action:
                    best_action = total_profit
                    best_rate = rate
                    best_profit = immediate_profit

        # Record the best rate and profit in the corresponding tracking dictionaries
        actions[(t, charge)] = best_rate
        profit_tracking[(t, charge)] = best_profit

        # Memoize the result for the current state
        memo[(t, charge)] = best_action

        return best_action


    max_profit = dp(0, 0)
    t, charge = 0, 0
    optimal_actions = []
    daily_profit = []

    while t < n:
        rate = actions[(t, charge)]
        optimal_actions.append(rate)
        daily_profit.append(profit_tracking[(t, charge)])
        charge += rate
        t += 1

    return max_profit, optimal_actions, daily_profit



prices = np.random.uniform(-2, 2, size=96)  # Random prices, can be negative
capacity = 2000 # Maximum battery capacity
max_rate = 2000 # Maximum charge/discharge rate
grid_limit = 2000 # Maximum rate at which we can buy/sell from/to the grid

start_time = time.time()
profit, optimal_actions, daily_profit = battery_optimization(prices, capacity, max_rate, grid_limit)
print(f"Execution time: {time.time() - start_time:.2f} seconds")
print("Maximum Profit:", profit)
#print("Optimal Actions:", optimal_actions)
# Print a table with the price and the action taken
print(f"{'Time Step':<10}{'Price':<10}{'Action':<10}{'Daily Profit':<15}")
for t in range(len(prices)):
    print(f"{t:<10}{prices[t]:<10.2f}{optimal_actions[t]:<10}{daily_profit[t]:<15.2f}")



# Plot the battery capacity over time
soc = [0]  # Start with an empty battery
for action in optimal_actions:
    new_soc = soc[-1] + action
    # Ensure SoC stays within 0 and capacity
    soc.append(min(max(new_soc, 0), capacity))

time_range = range(96)
plt.figure(figsize=(16, 12))

# Plot 1: Optimal Actions (Charging/Discharging Rates)
plt.subplot(3, 1, 1)
plt.bar(time_range, optimal_actions, color='orange', alpha=0.7, label='Optimal Charging/Discharging Rates')
plt.axhline(0, color='black', linestyle='--')
plt.title('Optimal Battery Charging/Discharging Rates')
plt.xlabel('Time (hours)')
plt.ylabel('Rate (kWh)')
plt.xticks(ticks=range(0, 96, 4), labels=range(0, 24))
plt.legend()
plt.grid(True)

# Plot 2: SOC over Time
plt.subplot(3, 1, 2)
plt.plot(time_range, soc[:-1], label='State of Charge (SOC)', marker='s', color='green')
plt.title('State of Charge Over Time')
plt.xlabel('Time (hours)')
plt.ylabel('SOC (kWh)')
plt.xticks(ticks=range(0, 96, 4), labels=range(0, 24))
plt.legend()
plt.grid(True)

# Plot 3: Daily Profit Plot
plt.subplot(3, 1, 3)
plt.bar(time_range, daily_profit, color='purple', alpha=0.7, label='Daily Profit')
cumulative_profit = np.cumsum(daily_profit)
plt.plot(time_range, cumulative_profit, label='Cumulative Profit', marker='o', color='blue')
plt.title('Daily and Cumulative Profit Over Time')
plt.xlabel('Time (hours)')
plt.ylabel('Profit ($)')
plt.xticks(ticks=range(0, 96, 4), labels=range(0, 24))
plt.legend()
plt.grid(True)

# Annotate the last point of the cumulative profit plot with its value
last_time_step = time_range[-1]
last_cumulative_profit = cumulative_profit[-1]
plt.annotate(f'{last_cumulative_profit:.2f}', xy=(last_time_step, last_cumulative_profit), xytext=(last_time_step, last_cumulative_profit + 10),
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10, ha='center')

# Adjust layout to prevent x-axis labels from overlapping
plt.tight_layout()

plt.show()
