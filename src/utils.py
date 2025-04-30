import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

def empirical_cdf(samples):
    """Compute the empirical CDF from samples."""
    sorted_samples = np.sort(samples)
    y = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
    return sorted_samples, y

def empirical_quantile(samples):
    """Compute the empirical quantile function from samples."""
    x, y = empirical_cdf(samples)
    return x, 1 - y

def link_function(cdf_value):
    """Link function for regular distributions: h(x; F) = 1/(1-F(x))"""
    return 1 / (1 - cdf_value)


def inverse_link_function(h_value):
    """Inverse of the link function: F(x) = 1 - 1/h(x)"""
    return 1 - 1/h_value


def compute_virtual_value(value, cdf_value, pdf_value):
    """Compute virtual value: φ(v) = v - (1-F(v))/f(v)"""
    if pdf_value <= 1e-10:  # Avoid division by zero
        return value
    return value - (1 - cdf_value) / pdf_value


def is_regular(values, cdf_values, pdf_values):
    """Check if a distribution is regular by testing if virtual values are non-decreasing"""
    virtual_values = [compute_virtual_value(v, F, f) for v, F, f in zip(values, cdf_values, pdf_values)]
    return all(virtual_values[i] <= virtual_values[i+1] for i in range(len(virtual_values)-1))


def convex_envelope(x_values, y_values):
    """
    Compute the convex envelope of a function, where the convex envelope is the
    maximum convex lower bound of the input function.
    That is, Conv(f) = sup{g(x) | g(x) is convex and g(x) <= f(x) for all x}
    
    Parameters:
    - x_values: array of x coordinates
    - y_values: array of y coordinates (function values)
    
    Returns:
    - y_envelope: array of convex envelope values
    """
    n = len(x_values)
    if n <= 2: # If there are 2 or fewer points, there's no concavity or convexity
        return y_values
        
    # Graham scan algorithm for convex hull
    hull = []
    for i in range(n):
        while len(hull) >= 2:
            # Check if the slope is decreasing
            slope1 = (y_values[hull[-1]] - y_values[hull[-2]]) / (x_values[hull[-1]] - x_values[hull[-2]])
            slope2 = (y_values[i] - y_values[hull[-1]]) / (x_values[i] - x_values[hull[-1]])
            if slope2 >= slope1:    # Maintain convexity
                # print(f"Removing point {hull[-1]} from hull due to slope violation")
                break
            hull.pop()  # Remove last point if convexity violated
        hull.append(i)
    
    # Create the convex envelope
    y_envelope = np.zeros_like(y_values)
    hull_indices = np.array(hull)
    
    # Linear interpolation between hull points
    for i in range(n):
        if i in hull_indices:
            y_envelope[i] = y_values[i]
        else:
            # Find the hull points that the current x is between
            right_idx = hull_indices[hull_indices > i].min()
            left_idx = hull_indices[hull_indices < i].max()
            
            # Linear interpolation
            t = (x_values[i] - x_values[left_idx]) / (x_values[right_idx] - x_values[left_idx])
            y_envelope[i] = y_values[left_idx] + t * (y_values[right_idx] - y_values[left_idx])
    
    return y_envelope


def compute_ks_distance(dist1, dist2, values):
    """Compute Kolmogorov-Smirnov distance between two distributions"""
    cdf1 = np.array([dist1.cdf(v) for v in values])
    cdf2 = np.array([dist2.cdf(v) for v in values])
    return np.max(np.abs(cdf1 - cdf2))


def optimal_reserve_price(distribution):
    """
    Find the optimal reserve price for a single-bidder auction
    
    For a regular distribution, this is the value v such that φ(v) = 0
    """
    def revenue(price):
        # Expected revenue is price * Pr[v >= price]
        return -price * distribution.quantile(price)  # Negative for minimization
    
    result = minimize_scalar(revenue, bounds=(0, 100), method='bounded')
    return result.x


def plot_revenue_vs_alpha(alphas, revenues, optimal_revenue, distribution_name):
    """
    Plot the revenue performance against alpha values.

    Parameters:
    - alphas: list of alpha values
    - revenues: list of learned revenues
    - optimal_revenue: optimal revenue for the distribution
    - distribution_name: name of the distribution for labeling

    """
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, [rev for rev in revenues], marker='o')
    plt.axhline(y=optimal_revenue, color='r', linestyle='--', label='Optimal Revenue')
    
    # Add theoretical bound if available (e.g., 1-O(√α) for regular distributions)

    # gaps = [(optimal_revenue - revenues) / (alpha * optimal_revenue)]
    if "regular" in distribution_name.lower():
        plt.plot(alphas, [(1 - np.sqrt(a)) * optimal_revenue for a in alphas], 'g--', label='Theoretical Bound (1-√α)')
    elif "mhr" in distribution_name.lower():
        plt.plot(alphas, [(1 - a) * optimal_revenue for a in alphas], 'g--', label='Theoretical Bound (1-α)')
    
    plt.title(f'Revenue Performance vs. Alpha ({distribution_name})')
    plt.xlabel('Alpha (α)')
    plt.ylabel('Revenue')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'plots/revenue_vs_alpha_{distribution_name.lower().replace(" ", "_")}.png')
    plt.close()


def plot_tester_robustness(distributions, alphas, results):
    """
    Create a heatmap showing when the tester says YES vs NO
    
    Parameters:
    - distributions: list of distribution names
    - alphas: list of alpha values
    - results: 2D array where results[i][j] is 1 if tester said YES for 
               distribution i with alpha j, and 0 otherwise
    """
    plt.figure(figsize=(12, 8))
    plt.imshow(results, cmap='viridis', aspect='auto')
    plt.colorbar(ticks=[0, 1], label='Tester Result (0=NO, 1=YES)')
    
    # Set tick labels
    plt.xticks(np.arange(len(distributions)), distributions, rotation=45)
    plt.yticks(np.arange(len(alphas)), [f'α={a}' for a in alphas])
    
    plt.title('Regularized Tester Robustness')
    plt.tight_layout()
    plt.savefig('tester_robustness.png')
    plt.close()

def plot_reserve_prices(alphas, learned_reserves, optimal_reserve, distribution_name):
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, learned_reserves, marker='o', label='Learned Reserve')
    plt.axhline(y=optimal_reserve, color='r', linestyle='--', label='Optimal Reserve')
    
    plt.title(f'Reserve Price vs. Alpha ({distribution_name})')
    plt.xlabel('Alpha (α)')
    plt.ylabel('Reserve Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'reserve_price_{distribution_name.lower().replace(" ", "_")}.png')
    plt.close()