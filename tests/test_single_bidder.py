import numpy as np
import matplotlib.pyplot as plt
from src.distributions import UniformDistribution, ProductDistribution, MixtureDistribution
from src.algorithms import regularized_tester, regularized_learner
from src.utils import optimal_reserve_price, plot_revenue_vs_alpha
from src.mechanisms import myerson_auction, vickrey_auction


# TODO
# Make more plots
# Benchmarks
# How robust is the tester to say no
# Regular distribution with epsilon that changes the regularity. More irregular as epsilon increases. As epsilon increases, how does the ratio change? Ratio of that tied to revenue
# Find ways to show that this is helpful. That the result is actually close to the optimal of the true distribution, not jsut a nearby one
# Theoretical vs actual revenue bounds relative to the optimal reserve price

# Set random seed for reproducibility
np.random.seed(22)

def test_single_bidder_uniform():
    """Test the algorithms on a single bidder with uniform distribution"""
    print("Testing Single Bidder with Uniform Distribution [0,1]")
    print("Theoretical result: Optimal reserve price = 0.5, Optimal revenue = 0.25")
    
    # Create uniform distribution
    dist = UniformDistribution(0, 1)
    
    # Generate samples
    num_samples = 1000
    samples = [dist.sample(num_samples)]

    
    # Test various alpha values
    alpha_values = [0.01, 0.05, 0.1, 0.2]
    revenues = [] # to store revenues for plotting, should map to alpha_values
    optimal_revenue = 0.25  # Theoretical optimal revenue for uniform distribution [0,1]
    
    for alpha in alpha_values:
        # Run tester
        tester_result = regularized_tester(samples, alpha)
        
        # Run learner
        reserve_prices = regularized_learner(samples, alpha)
        
        # Evaluate revenue through simulation
        revenue = evaluate_revenue(dist, reserve_prices)
        revenues.append(revenue)
        
        print(f"\nAlpha = {alpha}:")
        print(f"Tester result: {tester_result}")
        print(f"Learned reserve price: {reserve_prices[0]:.4f}")
        print(f"Estimated revenue: {revenue:.4f}")
        print(f"Approximation ratio: {revenue/0.25:.4f}")

    plot_revenue_vs_alpha(alpha_values, revenues, optimal_revenue, "Regular: Uniform Distribution [0,1]")


# def test_two_bidders_uniform():
#     """Test the algorithms on two i.i.d. bidders with uniform distribution"""
#     print("\nTesting Two I.I.D. Bidders with Uniform Distribution [0,1]")
#     print("Theoretical result: Optimal reserve price = 0.5, Optimal revenue ≈ 0.416")
    
#     # Create uniform distributions
#     dist1 = UniformDistribution(0, 1)
#     dist2 = UniformDistribution(0, 1)
#     product_dist = ProductDistribution([dist1, dist2])
    
#     # Generate samples
#     num_samples = 1000
#     samples = [dist1.sample(num_samples), dist2.sample(num_samples)]
    
#     # Test various alpha values
#     alpha_values = [0.01, 0.05, 0.1, 0.2]
    
#     for alpha in alpha_values:
#         # Run tester
#         tester_result = regularized_tester(samples, alpha)
        
#         # Run learner
#         reserve_prices = regularized_learner(samples, alpha)
        
#         # Evaluate revenue through simulation
#         revenue = evaluate_revenue_multi(product_dist, reserve_prices)
        
#         print(f"\nAlpha = {alpha}:")
#         print(f"Tester result: {tester_result}")
#         print(f"Learned reserve prices: {[round(p, 4) for p in reserve_prices]}")
#         print(f"Estimated revenue: {revenue:.4f}")
#         print(f"Approximation ratio: {revenue/0.416:.4f}")


def evaluate_revenue(distribution, reserve_prices, num_trials=10000):
    """Evaluate expected revenue through simulation for single bidder"""
    total_revenue = 0
    for _ in range(num_trials):
        bid = distribution.sample(1)[0]
        if bid >= reserve_prices[0]:
            total_revenue += reserve_prices[0]
    
    return total_revenue / num_trials


def test_single_bidder_irregular():
    """Test the algorithms on a single bidder with an irregular distribution"""

    # Learner should output yes when alpha is greater than epsilon
    
    # Create the irregular distribution (a mixture)
    dist1 = UniformDistribution(0, 1)
    dist2 = UniformDistribution(1048576, 1048577)
    dist = MixtureDistribution(dist1, dist2, p1=0.95)
    print("\nTesting Single Bidder with Irregular Distribution (Mixture)")
    print("This is a mixture of uniform[0,1] (with 95% probability) and uniform[1048576,1048577] (with 5% probability)")
    
    
    # Generate samples
    num_samples = 1000
    samples = [dist.sample(num_samples)]

    # Print min and max of samples for debugging
    # print(f"Samples min: {np.min(samples[0]):.4f}, max: {np.max(samples[0]):.4f}") # worked properly
    
    # Plot samples histogram, PDF, CDF, and quantile function
    # plot_distribution_diagnostics(dist, samples[0]) # looks right
    
    # Plot the virtual value function to verify irregularity
    verify_irregularity(dist) # looks right
    
    # Test various alpha values
    alpha_values = [0.001, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5]
    revenues = [] # to store revenues for plotting, should map to alpha_values

    # plot_revenue_vs_alpha(alpha_values)
    
    # Compute the theoretical optimal revenue
    # For this mixture, the optimal is to set reserve at 1048576, getting 1048576 * 0.05 = 52428.8
    optimal_revenue = 52428.8
    
    for alpha in alpha_values:
        # Run tester
        tester_result = regularized_tester(samples, alpha)
        
        # Run learner
        reserve_prices = regularized_learner(samples, alpha)
        
        # Evaluate revenue through simulation
        revenue = evaluate_revenue(dist, reserve_prices)
        revenues.append(revenue)
        
        print(f"\nAlpha = {alpha}:")
        print(f"Tester result: {tester_result}")
        print(f"Learned reserve price: {reserve_prices[0]:.4f}")
        print(f"Estimated revenue: {revenue:.4f}")
        print(f"Approximation ratio: {revenue/optimal_revenue:.4f}")
        
        # Also check what happens with the optimal reserve
        optimal_reserve = [1048576]
        optimal_sim_revenue = evaluate_revenue(dist, optimal_reserve)
        print(f"Revenue with optimal reserve price ({optimal_reserve[0]}): {optimal_sim_revenue:.4f}")

    plot_revenue_vs_alpha(alpha_values, revenues, optimal_revenue, "Irregular: Mixture of Uniform Distributions")

def plot_distribution_diagnostics(dist, samples, num_points=1000):
    """Plot various diagnostics for a distribution"""
    plt.figure(figsize=(20, 15))
    
    # Plot 1: Histogram of samples
    plt.subplot(2, 2, 1)
    plt.hist(samples, bins=50, density=True, alpha=0.7)
    plt.title('Histogram of Samples')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.grid(True)
    
    # For plotting continuous functions
    x_vals = np.linspace(0, 11, num_points)
    
    # Plot 2: PDF
    plt.subplot(2, 2, 2)
    pdf_vals = [dist.pdf(x) for x in x_vals]
    plt.plot(x_vals, pdf_vals)
    plt.title('Probability Density Function (PDF)')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.grid(True)
    
    # Plot 3: CDF
    plt.subplot(2, 2, 3)
    cdf_vals = [dist.cdf(x) for x in x_vals]
    plt.plot(x_vals, cdf_vals)
    plt.title('Cumulative Distribution Function (CDF)')
    plt.xlabel('Value')
    plt.ylabel('Probability')
    plt.grid(True)
    
    # Plot 4: Quantile Function
    plt.subplot(2, 2, 4)
    quantile_vals = [dist.quantile(x) for x in x_vals]
    plt.plot(x_vals, quantile_vals)
    plt.title('Quantile Function (q(x) = 1 - F(x))')
    plt.xlabel('Value')
    plt.ylabel('Quantile')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('plots/debugging/distribution_diagnostics.png')
    plt.close()

def verify_irregularity(dist, num_points=1000):
    """Plot the virtual value function to verify a distribution is irregular"""
    x_vals = np.linspace(0, 11, num_points)
    
    # Calculate virtual values
    virtual_values = []
    epsilon = 1e-10  # Small value to avoid division by zero
    
    for x in x_vals:
        pdf_val = max(dist.pdf(x), epsilon)
        virtual_val = x - (1 - dist.cdf(x)) / pdf_val
        virtual_values.append(virtual_val)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, virtual_values)
    plt.title('Virtual Value Function')
    plt.xlabel('Value')
    plt.ylabel('Virtual Value')
    plt.grid(True)
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='r', linestyle='-')
    
    # Save or show
    plt.savefig('plots/debugging/virtual_values_irregular.png')
    plt.close()
    
    # Check if it's non-monotonic (which would make it irregular)
    is_monotonic = all(virtual_values[i] <= virtual_values[i+1] for i in range(len(virtual_values)-1))
    print(f"Is the virtual value function monotonic? {is_monotonic}")
    if not is_monotonic:
        print("The distribution is irregular as expected.")
    else:
        print("Warning: The distribution might not be irregular!")\
        
# def plot_revenues_vs_regularity(epsilons):
#     """
#     Plot learned and theoretical revenues for various distributions of different regularity.
#     """

#     learned_revenues = [] # to store learned revenues
#     optimal_revenues = [] # to store theoretical revenues
#     for epsilon in epsilons:
#         # Create the distribution
#         dist1 = UniformDistribution(0, 1)
#         dist1_low = 2 ** (1 / epsilon)
#         dist1_high = dist1_low + 1
#         dist2 = UniformDistribution(dist1_low, dist1_high)
#         dist = MixtureDistribution(dist1, dist2, p1=(1-epsilon))

#         # PRINT DISTRIBUTION INFORMATION
#         print(f"\nDistribution with epsilon={epsilon}:")
#         print(f"Uniform from 0 to 1 with probability {1-epsilon} and uniform from {dist1_low} to {dist1_high} with probability {epsilon}")
        
#         # Generate samples
#         num_samples = 1000
#         samples = [dist.sample(num_samples)]

#         # Variety of alpha values
#         alpha_values = [0.01, 0.05, 0.1, 0.2, 0.5]

#         for alpha in alpha_values:
#             # Run tester
#             tester_result = regularized_tester(samples, alpha)
            
#             # Run learner
#             reserve_prices = regularized_learner(samples, alpha)
            
#             # Evaluate revenue through simulation
#             revenue = evaluate_revenue(dist, reserve_prices)
#             learned_revenues.append(revenue)

#             # Also check what happens with the optimal reserve
#             optimal_reserve = [dist1_low]
#             optimal_sim_revenue = evaluate_revenue(dist, optimal_reserve)
#             optimal_revenue = optimal_reserve_price * epsilon
#             optimal_revenues.append(optimal_revenue)

#             print(f"\nAlpha = {alpha}:")
#             print(f"Tester result: {tester_result}")
#             print(f"Learned reserve price: {reserve_prices[0]:.4f}")
#             print(f"Estimated revenue: {revenue:.4f}")
#             print(f"Approximation ratio: {revenue/optimal_revenue:.4f}")
#             print(f"Simulated revenue with optimal reserve price ({optimal_reserve[0]}): {optimal_sim_revenue:.4f}")
#             print(f"Theoretical optimal revenue: {optimal_revenue:.4f}")

def plot_revenues_vs_regularity(a=0, b=1, epsilons=[0.01, 0.05, 0.1, 0.2, 0.3], alpha_values=[0.01, 0.05, 0.1, 0.2, 0.3]):
    """
    Plot learned and theoretical revenues for various distributions of different regularity.
    
    Parameters:
    - a: Lower bound for the second uniform distribution
    - b: Upper bound for the second uniform distribution
    - epsilons: List of epsilon values to test (higher epsilon = more irregular)
    - alpha_values: List of alpha values to test for the regularized tester/learner
    """
    fig, axs = plt.subplots(2, 2, figsize=(20, 16))
    axs = axs.flatten()
    
    # Data structures to store results
    all_learned_revenues = []  # [epsilon][alpha]
    all_optimal_revenues = []  # [epsilon]
    all_optimal_sim_revenues = []  # [epsilon]
    all_tester_results = []  # [epsilon][alpha]
    all_reserve_prices = []  # [epsilon][alpha]
    all_approximation_ratios = []  # [epsilon][alpha]
    
    # Create a colormap for the different alpha values
    colors = plt.cm.viridis(np.linspace(0, 1, len(alpha_values)))
    
    for epsilon in epsilons:
        # Create the distribution
        dist1 = UniformDistribution(0, 1)
        # For very small epsilon, cap the value to avoid numerical issues

        safe_epsilon = max(epsilon, 0.001)
        dist2_low = 2 ** (1 / safe_epsilon)
        # dist2_low = (1 / safe_epsilon) ** 2
        dist2_high = dist2_low + 1
        
        # dist2 = UniformDistribution(dist1_low, dist1_high)
        # dist2_low = a
        # dist2_high = b
        dist2 = UniformDistribution(dist2_low, dist2_high)
        dist = MixtureDistribution(dist1, dist2, p1=(1-epsilon))

        # PRINT DISTRIBUTION INFORMATION
        print(f"\nDistribution with epsilon={epsilon}:")
        print(f"Uniform from 0 to 1 with probability {1-epsilon} and uniform from {dist2_low:.2f} to {dist2_high:.2f} with probability {epsilon}")
        
        np.random.seed(42)
        # Generate samples
        num_samples = 1000
        samples = [dist.sample(num_samples)]
        
        # Calculate theoretical optimal revenue
        # For this mixture, optimal is to price at dist1_low, getting dist1_low * epsilon revenue
        # optimal_revenue = dist1_low * epsilon
        if epsilon == 0:
            optimal_revenue = 0.25
        else:
            optimal_revenue = dist2_low * epsilon
        all_optimal_revenues.append(optimal_revenue)
        
        # Also calculate the optimal revenue via simulation
        optimal_reserve = [dist2_low]
        optimal_sim_revenue = evaluate_revenue(dist, optimal_reserve)
        all_optimal_sim_revenues.append(optimal_sim_revenue)
        
        learned_revenues_for_epsilon = []
        tester_results_for_epsilon = []
        reserve_prices_for_epsilon = []
        approximation_ratios_for_epsilon = []

        for alpha in alpha_values:
            # Run tester
            tester_result = regularized_tester(samples, alpha)
            tester_results_for_epsilon.append(1 if tester_result == "YES" else 0)
            
            # Run learner
            reserve_prices = regularized_learner(samples, alpha)
            reserve_prices_for_epsilon.append(reserve_prices[0])
            
            # Evaluate revenue through simulation
            revenue = evaluate_revenue(dist, reserve_prices)
            learned_revenues_for_epsilon.append(revenue)
            
            # Calculate approximation ratio
            approximation_ratio = revenue / optimal_revenue
            approximation_ratios_for_epsilon.append(approximation_ratio)

            print(f"\nAlpha = {alpha}:")
            print(f"Tester result: {tester_result}")
            print(f"Learned reserve price: {reserve_prices[0]:.4f}")
            print(f"Estimated revenue: {revenue:.4f}")
            print(f"Approximation ratio: {approximation_ratio:.4f}")
            print(f"Simulated revenue with optimal reserve price ({optimal_reserve[0]:.4f}): {optimal_sim_revenue:.4f}")
            print(f"Theoretical optimal revenue: {optimal_revenue:.4f}")
        
        all_learned_revenues.append(learned_revenues_for_epsilon)
        all_tester_results.append(tester_results_for_epsilon)
        all_reserve_prices.append(reserve_prices_for_epsilon)
        all_approximation_ratios.append(approximation_ratios_for_epsilon)

    # Convert to numpy arrays for easier plotting
    all_learned_revenues = np.array(all_learned_revenues)
    all_optimal_revenues = np.array(all_optimal_revenues)
    all_optimal_sim_revenues = np.array(all_optimal_sim_revenues)
    all_tester_results = np.array(all_tester_results)
    all_reserve_prices = np.array(all_reserve_prices)
    all_approximation_ratios = np.array(all_approximation_ratios)
    
    # PLOT 1: Revenue vs Epsilon for different alpha values
    ax = axs[0]
    for i, alpha in enumerate(alpha_values):
        ax.plot(epsilons, all_learned_revenues[:, i], marker='o', color=colors[i], 
                label=f'α = {alpha}')
    
    # Plot the optimal revenue
    ax.plot(epsilons, all_optimal_revenues, 'k--', linewidth=2, label='Theoretical Optimal')
    # ax.plot(epsilons, all_optimal_sim_revenues, 'r:', linewidth=2, label='Simulated Optimal')
    
    ax.set_title('Revenue vs Epsilon (Degree of Irregularity)')
    ax.set_xlabel('Epsilon (higher = more irregular)')
    ax.set_ylabel('Revenue')
    ax.legend()
    ax.grid(True)
    
    # PLOT 2: Approximation Ratio vs Epsilon for different alpha values
    ax = axs[1]
    for i, alpha in enumerate(alpha_values):
        ax.plot(epsilons, all_approximation_ratios[:, i], marker='o', color=colors[i], 
                label=f'α = {alpha}')
    
    # Add theoretical bounds for reference
    for i, alpha in enumerate(alpha_values):
        ax.plot(epsilons, [1 - np.sqrt(alpha)] * len(epsilons), '--', color=colors[i], alpha=0.5,
                label=f'Theoretical bound for α={alpha}')
    
    ax.set_title('Approximation Ratio vs Epsilon')
    ax.set_xlabel('Epsilon (higher = more irregular)')
    ax.set_ylabel('Approximation Ratio (Learned/Optimal)')
    ax.axhline(y=1.0, color='k', linestyle='-', alpha=0.3)
    ax.legend()
    ax.grid(True)
    
    # PLOT 3: Reserve Price vs Epsilon for different alpha values
    ax = axs[2]
    for i, alpha in enumerate(alpha_values):
        ax.plot(epsilons, all_reserve_prices[:, i], marker='o', color=colors[i], 
                label=f'α = {alpha}')
    
    # Plot the theoretical optimal reserve price
    optimal_reserves = [2**(1/max(e, 0.001)) for e in epsilons]
    ax.plot(epsilons, optimal_reserves, 'k--', linewidth=2, label='Optimal Reserve')
    
    ax.set_title('Reserve Price vs Epsilon')
    ax.set_xlabel('Epsilon (higher = more irregular)')
    ax.set_ylabel('Reserve Price')
    ax.set_yscale('log')  # Log scale for better visualization
    ax.legend()
    ax.grid(True)
    
    # PLOT 4: Tester Results Heatmap
    ax = axs[3]
    im = ax.imshow(all_tester_results, cmap='viridis', aspect='auto')
    ax.set_title('Tester Results (YES=1, NO=0)')
    ax.set_xlabel('Alpha Value Index')
    ax.set_ylabel('Epsilon Value Index')
    ax.set_xticks(np.arange(len(alpha_values)))
    ax.set_yticks(np.arange(len(epsilons)))
    ax.set_xticklabels([f'{a}' for a in alpha_values])
    ax.set_yticklabels([f'{e}' for e in epsilons])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.set_ticklabels(['NO', 'YES'])
    
    # Add text annotations to the heatmap
    for i in range(len(epsilons)):
        for j in range(len(alpha_values)):
            text = ax.text(j, i, all_tester_results[i, j],
                          ha="center", va="center", color="w")
    
    plt.tight_layout()
    plt.savefig('revenue_vs_regularity.png', dpi=300)
    plt.show()
    
    # Additional specialized plot: Revenue ratio vs alpha for different epsilons
    plt.figure(figsize=(10, 8))
    for i, epsilon in enumerate(epsilons):
        plt.plot(alpha_values, all_approximation_ratios[i, :], marker='o', 
                 label=f'ε = {epsilon}')
    
    # Add theoretical bound line
    plt.plot(alpha_values, [1 - np.sqrt(a) for a in alpha_values], 'k--', 
             label='Theoretical bound (1-√α)')
    
    plt.title('Revenue Ratio vs Alpha for Different Levels of Irregularity')
    plt.xlabel('Alpha Value')
    plt.ylabel('Revenue Ratio (Learned/Optimal)')
    plt.legend()
    plt.grid(True)
    plt.savefig('plots/revenue_ratio_vs_alpha.png', dpi=300)
    plt.show()
    
    # Return the collected data for further analysis if needed
    return {
        'epsilons': epsilons,
        'alpha_values': alpha_values,
        'learned_revenues': all_learned_revenues,
        'optimal_revenues': all_optimal_revenues,
        'optimal_sim_revenues': all_optimal_sim_revenues,
        'tester_results': all_tester_results,
        'reserve_prices': all_reserve_prices,
        'approximation_ratios': all_approximation_ratios
    } 
     


    



if __name__ == "__main__":
    # test_single_bidder_uniform()
    # test_single_bidder_irregular()
    plot_revenues_vs_regularity(100, 101)