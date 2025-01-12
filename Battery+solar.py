import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
np.random.seed(10)

def battery_solar_optimization(prices, solar, capacity, max_rate, grid_limit):
    n = len(prices)
    min_soc = int(0.4 * capacity)  # Minimum SoC at the end of the day
    max_soc = int(0.6 * capacity)  # Maximum SoC at the end of the day
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
        best_solar_sell = 0
        best_profit = 0
        
        if solar[t] != 0:
            step =int(round(solar[t]/100))
            if step == 0:
                step = 1
            # Iterate over possible rates and solar selling amounts
            for rate in range(-max_rate, max_rate + 1, 10):  # Battery rates
                for sell_solar in range(0, round(solar[t]), step):  # Discretize solar selling
                    new_charge = charge + rate

                    # Ensure valid transitions
                    if 0 <= new_charge <= capacity and rate + sell_solar <= grid_limit:
                        # Calculate profits
                        solar_profit = sell_solar * prices[t]  # Selling solar
                        if rate > 0 and prices[t] < 0:  # Charging at negative prices
                            battery_profit = abs(rate) * abs(prices[t])
                        elif rate < 0 and prices[t] > 0:  # Discharging at positive prices
                            battery_profit = abs(rate) * prices[t]
                        else:  # Do nothing or invalid rate
                            battery_profit = -(rate * prices[t])

                        immediate_profit = solar_profit + battery_profit

                        # Recursive call to calculate the total profit
                        total_profit = immediate_profit + dp(t + 1, new_charge)

                        # Update the best action if this combination is optimal
                        if total_profit > best_action:
                            best_action = total_profit
                            best_rate = rate
                            best_solar_sell = sell_solar
                            best_profit = immediate_profit
        else:
            for rate in range(-max_rate, max_rate+1,10):
                new_charge = charge + rate
                sell_solar = 0

                # Ensure valid transitions
                if 0 <= new_charge <= capacity and rate <= grid_limit:
                    if rate > 0 and prices[t] < 0:  # Charging at negative prices
                        battery_profit = abs(rate) * abs(prices[t])
                    elif rate < 0 and prices[t] > 0:  # Discharging at positive prices
                        battery_profit = abs(rate) * prices[t]
                    else:  # Do nothing or invalid rate
                        battery_profit = -(rate * prices[t])

                    immediate_profit = battery_profit

                    # Recursive call to calculate the total profit
                    total_profit = immediate_profit + dp(t + 1, new_charge)

                    # Update the best action if this combination is optimal
                    if total_profit > best_action:
                        best_action = total_profit
                        best_rate = rate
                        best_solar_sell = sell_solar
                        best_profit = immediate_profit

        # Record the best decisions
        actions[(t, charge)] = (best_rate, best_solar_sell)
        profit_tracking[(t, charge)] = best_profit

        # Memoize the result for the current state
        memo[(t, charge)] = best_action
        return best_action

    max_profit = dp(0, 0)
    t, charge = 0, 0
    optimal_actions = []
    daily_profit = []

    while t < n:
        rate, sell_solar = actions[(t, charge)]
        optimal_actions.append((rate, sell_solar))
        daily_profit.append(profit_tracking[(t, charge)])
        charge += rate
        t += 1

    return max_profit, optimal_actions, daily_profit


# Random data for testing
prices = np.random.uniform(-2, 2, size=96)  # Random prices, can be negative
capacity = 2000  # Maximum battery capacity
max_rate = 2000  # Maximum charge/discharge rate
grid_limit = 2000  # Maximum rate at which we can buy/sell from/to the grid
pv_prod = pd.read_excel('pv_prodction.xlsx', usecols=['Return delivery (kWh)'])
solar_1 = pv_prod['Return delivery (kWh)'].values

solar=np.zeros(96)
aux=0

for i in range(10749,10845, 1):
    solar[aux]=solar_1[i]
    aux+=1


start_time = time.time()
profit, optimal_actions, daily_profit = battery_solar_optimization(prices, solar, capacity, max_rate, grid_limit)
print(f"Execution time: {time.time() - start_time:.2f} seconds")
print("Maximum Profit:", profit)

print(f"{'Time Step':<10}{'Price':<10}{'Solar':<10}{'Action (Rate, Sell Solar)':<25}{'Daily Profit':<15}")
for t in range(len(prices)):
    print(f"{t:<10}{prices[t]:<10.2f}{solar[t]:<10.2f}{str((optimal_actions[t][0], optimal_actions[t][1])):<25}{daily_profit[t]:<15.2f}")


# Plot the battery capacity and solar allocation over time
soc = [0]  # Start with an empty battery
solar_sell = [0]  # Solar sold to the grid

for action in optimal_actions:
    rate, sell_solar = action
    new_soc = soc[-1] + rate
    soc.append(min(max(new_soc, 0), capacity))
    solar_sell.append(sell_solar)

time_range = range(96)
plt.figure(figsize=(16, 12))

# Plot 1: Optimal Actions (Charging/Discharging, Selling Solar, and Solar Production)
plt.subplot(3, 1, 1)
plt.bar(time_range, [a[0] for a in optimal_actions], color='orange', alpha=0.7, label='Battery Rate')
plt.plot(time_range, solar_sell[1:], label='Solar Sold to Grid', marker='o', color='blue')
plt.plot(time_range, solar, label='Solar Production', marker='x', color='red')
plt.axhline(0, color='black', linestyle='--')
plt.title('Optimal Battery Charging/Discharging, Solar Selling, and Solar Production')
plt.xlabel('Time (hours)')
plt.ylabel('Rate (kWh)')
plt.legend()
plt.grid(True)

# Plot 2: SOC over Time
plt.subplot(3, 1, 2)
plt.plot(time_range, soc[:-1], label='State of Charge (SOC)', marker='s', color='green')
plt.title('State of Charge Over Time')
plt.xlabel('Time (hours)')
plt.ylabel('SOC (kWh)')
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
plt.legend()
plt.grid(True)

# Show the last value of the cumulative profit on the plot
last_value = cumulative_profit[-1]
plt.annotate(f'Total profit: {last_value:.2f}', xy=(time_range[-1], cumulative_profit[-1]), 
             xytext=(time_range[-1], cumulative_profit[-1] + 10),
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=12, color='blue')

# Adjust layout
plt.tight_layout()
plt.show()
