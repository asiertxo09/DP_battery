# Dynamic Programming for Battery Optimization

This repository contains two Python scripts that demonstrate the use of dynamic programming to optimize battery operation in different scenarios:

## Standalone_battery.py

This script simulates a standalone battery system operating under dynamic electricity pricing. It uses a value iteration approach to determine the optimal charging and discharging strategy to maximize profit while considering grid imbalance penalties.

**Key features:**

*   Dynamic pricing function that adjusts electricity prices based on grid balance.
*   Reward function that considers charging costs, discharging revenue, and imbalance penalties.
*   State transitions that simulate the battery's state of charge (SOC) over time.
*   Visualization of optimal actions, SOC, and profit.

**How to run:**

1.  Ensure you have Python 3 installed.
2.  Install the required libraries: `matplotlib`.
3.  Run the script: `python Standalone_battery.py`

## Battery+solar.py

This script extends the `Standalone_battery.py` by incorporating solar (PV) production. It optimizes the combined operation of the battery and solar panels, determining the best strategy for charging/discharging the battery and selling solar energy to the grid.

**Key features:**

*   Incorporates solar PV production into the optimization model.
*   Considers battery SOC, maximum charge/discharge rates, and grid connection limits.
*   Uses memoization to speed up the optimization process.
*   Visualizes optimal actions, SOC, solar production, and profit.

**How to run:**

1.  Ensure you have Python 3 installed.
2.  Install the required libraries: `matplotlib`.
3.  Run the script: `python solar_2.py`

## Results

The results of both simulations are visualized in the form of plots that show:

*   Optimal charging/discharging rates
*   State of charge over time
*   Daily and cumulative profit

These results demonstrate the effectiveness of dynamic programming in optimizing battery systems for energy arbitrage and grid balancing. The inclusion of solar PV generation further enhances the system's profitability and sustainability.

## Future Work

*   Incorporate battery degradation into the models.
*   Investigate the impact of battery operation on the grid.
*   Compare the performance of dynamic programming with other optimization techniques.
*   Extend the models to consider more complex scenarios, such as multiple batteries or demand response programs.

