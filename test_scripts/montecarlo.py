import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
win_rate = 0.56
odds = -115
payout_ratio = 100 / 115  # $0.87 payout 
initial_bankroll = 1000  
bet_size = 10  
num_bets = 3000  # Bets per season ~ 3 bets per game
num_simulations = 10000  

# Function to simulate a single betting session
def simulate_session(bankroll, bets, bet_size, win_rate, payout_ratio):
    for _ in range(bets):
        if np.random.rand() < win_rate:  # Win
            bankroll += bet_size * payout_ratio
        else:  # Lose
            bankroll -= bet_size
    return bankroll

# Run Monte Carlo simulations
final_bankrolls = []
for _ in range(num_simulations):
    final_bankroll = simulate_session(initial_bankroll, num_bets, bet_size, win_rate, payout_ratio)
    final_bankrolls.append(final_bankroll)

# Analyze results
final_bankrolls = np.array(final_bankrolls)
mean_bankroll = np.mean(final_bankrolls)
median_bankroll = np.median(final_bankrolls)
probability_profit = np.mean(final_bankrolls > initial_bankroll)

# Print results
print(f"Mean final bankroll: ${mean_bankroll:.2f}")
print(f"Median final bankroll: ${median_bankroll:.2f}")
print(f"Probability of being profitable: {probability_profit * 100:.2f}%")

# Plot histogram of results
plt.hist(final_bankrolls, bins=50, color='blue', alpha=0.7, edgecolor='black')
plt.axvline(initial_bankroll, color='red', linestyle='--', label='Initial Bankroll')
plt.title('Monte Carlo Simulation of NBA Betting')
plt.xlabel('Final Bankroll ($)')
plt.ylabel('Frequency')
plt.legend()
plt.show()
