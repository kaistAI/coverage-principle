import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import pandas as pd

# Input data
E = np.array([30, 50, 70, 100, 150, 200])
# N = np.array([30000, 150000, 382000, 1100000, 3800000, 8900000]) # Threehop - 4e-4
# N = np.array([30000, 150000, 382000, 1100000, 3800000, 8900000]) # Threehop - 8e-4
N = np.array([4000, 15000, 30000, 80000, 250000, 500000]) # Twohop - 8e-4

NAME = "twohop_8e-4"    

# Create a DataFrame for easier data handling
df = pd.DataFrame({'E': E, 'N': N})
print("Original Data:")
print(df)

# Function to compute power law
def power_law(x, a, b):
    return a * np.power(x, b)

# Fit the power law using non-linear least squares
params, covariance = optimize.curve_fit(power_law, E, N)
a, b = params
print(f"\nPower Law Model: N = {a:.4f} × E^{b:.4f}")

# Calculate R-squared in original space
N_fit = power_law(E, a, b)
ss_res = np.sum((N - N_fit) ** 2)
ss_tot = np.sum((N - np.mean(N)) ** 2)
r_squared = 1 - (ss_res / ss_tot)
print(f"R-squared (original space): {r_squared:.6f}")

# Also compute R-squared in log-log space
logE = np.log(E)
logN = np.log(N)
logN_fit = np.log(power_law(E, a, b))
log_ss_res = np.sum((logN - logN_fit) ** 2)
log_ss_tot = np.sum((logN - np.mean(logN)) ** 2)
log_r_squared = 1 - (log_ss_res / log_ss_tot)
print(f"R-squared (log-log space): {log_r_squared:.6f}")

# Calculate errors for each data point
df['N_fitted'] = power_law(df['E'], a, b)
df['Error'] = np.abs(df['N'] - df['N_fitted']) / df['N'] * 100
print("\nComparison of actual vs. fitted values:")
print(df[['E', 'N', 'N_fitted', 'Error']].to_string(float_format=lambda x: f'{x:.1f}'))

# Create smooth curve for plotting
E_smooth = np.linspace(min(E), max(E), 100)
N_smooth = power_law(E_smooth, a, b)

# Plotting
plt.figure(figsize=(12, 10))

# Plot 1: Original data with power law fit
plt.subplot(2, 2, 1)
plt.scatter(E, N, color='blue', label='Data points')
plt.plot(E_smooth, N_smooth, 'r-', label=f'Fit: N = {a:.4f} × E^{b:.4f}')
plt.xlabel('E')
plt.ylabel('N')
plt.title('Power Law Fit')
plt.legend()
plt.grid(True)

# Plot 2: Log-log plot 
plt.subplot(2, 2, 2)
plt.scatter(np.log10(E), np.log10(N), color='blue', label='Data points')
# plt.scatter(np.log10(E[:3]), np.log10(N[:3]), color='blue', label='Data points')
# plt.scatter(np.log10(E[3:]), np.log10(N[3:]), color='red', label='Data points(expected)')
plt.plot(np.log10(E_smooth), np.log10(N_smooth), 'r-', label=f'Fit: log(N) = log({a:.4f}) + {b:.4f}log(E)')
plt.xlabel('log(E)')
plt.ylabel('log(N)')
plt.title('Power Law Fit in Log-Log Space')
plt.legend()
plt.grid(True)

# Plot 3: Residuals
plt.subplot(2, 2, 3)
plt.scatter(E, (N - N_fit), color='green')
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('E')
plt.ylabel('Residuals (N - N_fit)')
plt.title('Residuals')
plt.grid(True)

# Plot 4: Percentage Error
plt.subplot(2, 2, 4)
plt.bar(E, df['Error'], width=10)
plt.xlabel('E')
plt.ylabel('Error (%)')
plt.title('Percentage Error')
plt.grid(True)

plt.tight_layout()
plt.savefig(f'{NAME}_power_law_fit.png')
plt.show()

# Additional verification: Print some extrapolated values
print("\nExtrapolated values:")
for e_val in [25, 75, 120, 175, 225]:
    predicted = power_law(e_val, a, b)
    print(f"E = {e_val}: Predicted N = {predicted:.0f}")