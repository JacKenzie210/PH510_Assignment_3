import numpy as np
import matplotlib.pyplot as plt

def monte_carlo_integral_subinterval(func, a, x, num_samples=1000):
    """
    Estimates the integral of `func` from a to x using Monte Carlo integration.
    
    Parameters:
        func (callable): The function to integrate.
        a (float): Lower bound of integration.
        x (float): Upper bound of integration.
        num_samples (int): Number of random samples.
    
    Returns:
        float: Estimated integral from a to x.
    """
    if x == a:
        return 0.0
    # Generate random sample points in [a, x]
    samples = np.random.uniform(a, x, num_samples)
    # Monte Carlo estimate: (x - a) * mean(func(samples))
    return (x - a) * np.mean(func(samples))

# Define the function to integrate
def f(x):
    return np.sin(x)

# Integration bounds
a = 0
b =2* np.pi

# Create a grid of x values over [a, b]
num_grid = 100
x_grid = np.linspace(a, b, num_grid)

# Estimate the antiderivative at each x value using Monte Carlo integration
F_est = np.array([monte_carlo_integral_subinterval(f, a, xi, num_samples=1000) for xi in x_grid])

# The true antiderivative of sin(x) is -cos(x), adjusting constant so that F(a)=0:
F_true = -np.cos(x_grid) + np.cos(a)

# Plotting the results
plt.figure(figsize=(8, 6))
plt.plot(x_grid, F_est, label="Monte Carlo Estimate", color="blue")
plt.plot(x_grid, F_true, label="True Antiderivative", color="red", linestyle="--")
plt.xlabel("x")
plt.ylabel("F(x)")
plt.title("Antiderivative of sin(x) using Monte Carlo Integration")
plt.legend()
plt.show()
