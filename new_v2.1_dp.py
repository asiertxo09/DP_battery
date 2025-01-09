import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
np.random.seed(10)

def battery_optimization(prices, capacity, grid_limit, pv):
    n = len(prices)
    min_soc = 40
    max_soc = 60
    rates = np.linspace(-10, 10, 3)  
    memo = {}
    actions = {}
    profit_tracking = {}

    def dp(t, charge):
        # Base case with corrected index check
        if t >= n:
            return 0 if min_soc <= charge <= max_soc else float('-inf')

        # Memoization check
        if (t, charge) in memo:
            return memo[(t, charge)]

        # Initialize variables for optimization
        best_action = float('-inf')
        best_rate = 0
        best_profit = 0
        best_charge_memo = int(round(charge * 100))

        for rate in rates:
            new_charge = charge + rate
            new_charge_memo = int(round(new_charge*100))



            if 0 <= new_charge <= 100 and rate + pv[t] <= grid_limit:
                
                # **Sell full PV production and charge from grid**
                pv_sell = pv[t] * prices[t]
                if rate > 0 and prices[t] < 0:  # Charging at negative prices
                    immediate_profit = abs(rate) * prices[t]
                elif rate < 0 and prices[t] > 0:  # Discharging at positive prices
                    immediate_profit = abs(rate) * prices[t]
                else:  # Do nothing or invalid rate
                    immediate_profit = -(rate * prices[t])

                if (t+1, new_charge) in memo:
                
                    total_profit1 = immediate_profit + pv_sell + memo[(t+1, new_charge_memo)]
                else:
                    total_profit1 = immediate_profit + pv_sell + dp(t + 1, new_charge)

                # **Sell part of PV and charge battery with the rest**
                best_partial_profit = float('-inf')
                for pv_s in np.linspace(0, pv[t], 2):

                    pv_sell_partial = pv_s * prices[t]
                    new_charge_partial = charge + rate + ((pv[t] - pv_s)/20)
                    new_charge_partial_memo = int(round(new_charge_partial*100))
                    
                    if 0 <= new_charge_partial <= 100:
                        if rate > 0 and prices[t] < 0:  # Charging at negative prices
                            immediate_profit_partial = abs(rate) * prices[t]
                        elif rate < 0 and prices[t] > 0:  # Discharging at positive prices
                            immediate_profit_partial = abs(rate) * prices[t]
                        else:  # Do nothing or invalid rate
                            immediate_profit_partial = -(rate * prices[t])

                        if (t+1, new_charge_partial_memo) in memo:
                
                            total_profit_partial = immediate_profit_partial + pv_sell_partial + memo[(t+1, new_charge_partial_memo)]
                        else:
                            total_profit_partial = immediate_profit_partial + pv_sell_partial + dp(t + 1, new_charge_partial)
                            
                        if total_profit_partial > best_partial_profit:
                            best_partial_profit = total_profit_partial
            
                # **Charge fully with PV and sell rest**
                new_charge_full = charge + pv[t]/20

                if (100-new_charge_full)>=rate:
                    if rate > 0 and prices[t] < 0:  # Charging at negative prices
                        immediate_profit_3 = abs(rate) * prices[t]
                    elif rate < 0 and prices[t] > 0:  # Discharging at positive prices
                        immediate_profit_3 = abs(rate) * prices[t]
                    else:  # Do nothing or invalid rate
                        immediate_profit_3 = -(rate * prices[t])

                    new_charge_full += rate

                new_charge_full_memo = int(round(new_charge_full*100))

                if (t+1, new_charge_full) in memo:
                
                    total_profit3 = immediate_profit_3  + memo[(t+1, new_charge_full_memo)]
                else:
                    total_profit3= immediate_profit_3 + dp(t + 1, new_charge_full)

                # Compare profits
                total_profit = max(total_profit1,best_partial_profit,  total_profit3)

                if total_profit1 == total_profit:
                    if total_profit > best_action:
                        best_action = total_profit
                        best_rate = rate
                        best_profit = immediate_profit
                        best_charge_memo = int(round(new_charge,2)*100)
                elif total_profit_partial == total_profit:
                    if total_profit > best_action:
                        best_action = total_profit
                        best_rate = rate
                        best_profit = immediate_profit_partial
                        best_charge_memo = int(round(new_charge_partial,2)*100)
                else:
                    if total_profit > best_action:
                        best_action = total_profit
                        best_rate = rate
                        best_profit = immediate_profit_3
                        best_charge_memo = int(round(new_charge_full,2)*100)
                    

        # Store results in memoization
        actions[(t, best_charge_memo)] = best_rate
        profit_tracking[(t, best_charge_memo)] = best_profit
        memo[(t, best_charge_memo)] = best_action
        return best_action

    # Compute optimal profit
    max_profit = dp(0, 0)
    t, charge = 0, 0
    optimal_actions = []
    daily_profit = []
    soc = [charge]

    while t < n:
        charge_memo = int(round(charge * 100))
        rate = actions[(t, charge_memo)]
        optimal_actions.append(rate)
        daily_profit.append(profit_tracking[(t, charge_memo)])
        charge += rate
        soc.append(charge)
        t += 1

    return max_profit, optimal_actions, daily_profit, soc

# **Data Input**
start_time = time.time()
prices = np.random.uniform(-2, 2, size=24)  
capacity = 2000
grid_limit = 2000

# **Load and Validate PV Production Data**
pv_prod = pd.read_excel('pv_prodction.xlsx', usecols=['Return delivery (kWh)'])
pv = pv_prod['Return delivery (kWh)'].values

pv_2=np.zeros(24)
aux=0

for i in range(10749,10845, 4):
    for j in range(4):
        pv_2[aux]+=pv[i+j]
    pv_2[aux]=pv_2[aux]/4
    aux+=1


print("start")
profit, optimal_actions, daily_profit, soc = battery_optimization(prices, capacity, grid_limit, pv_2)
print("Maximum Profit:", profit)
print(f"Execution time: {time.time() - start_time:.2f} seconds")

# **Plots**
time_range = range(24)

plt.figure(figsize=(12, 8))

# Plot 1: Prices and PV Production
plt.subplot(4, 1, 1)
plt.plot(time_range, prices, label='Electricity Prices', marker='o')
plt.plot(time_range, pv_2, label='PV Production', marker='x')
plt.title('Electricity Prices and PV Production')
plt.xlabel('Time (hours)')
plt.ylabel('Values')
plt.legend()
plt.grid(True)

# Plot 2: Optimal Actions (Charging/Discharging Rates)
plt.subplot(4, 1, 2)
plt.bar(time_range, optimal_actions, color='orange', alpha=0.7, label='Optimal Charging/Discharging Rates')
plt.axhline(0, color='black', linestyle='--')
plt.title('Optimal Battery Charging/Discharging Rates')
plt.xlabel('Time (hours)')
plt.ylabel('Rate (kWh)')
plt.legend()
plt.grid(True)

# Plot 3: SOC over Time
plt.subplot(4, 1, 3)
plt.plot(time_range, soc[:-1], label='State of Charge (SOC)', marker='s', color='green')
plt.title('State of Charge Over Time')
plt.xlabel('Time (hours)')
plt.ylabel('SOC (kWh)')
plt.legend()
plt.grid(True)


# Daily Profit Plot
plt.subplot(4, 1, 4)
plt.figure(figsize=(8, 5))
plt.bar(time_range, daily_profit, color='purple', alpha=0.7)
plt.title('Daily Profit Over Time')
plt.xlabel('Time (hours)')
plt.ylabel('Profit ($)')
plt.grid(True)
plt.show()

