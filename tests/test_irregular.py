import numpy as np
import matplotlib.pyplot as plt
from src.distributions import UniformDistribution, ProductDistribution, MixtureDistribution, GaussianDistribution, LogNormalDistribution
from src.algorithms import regularized_tester, regularized_learner
from src.utils import link_function, inverse_link_function, convex_envelope


def debug_specific_case(epsilon=0.1, alpha=0.01, num_samples=1000, seed=22, delta=0.05, given_dist=None):
    """
    Debug specific cases where the tester behaves unexpectedly.
    Visualizes the link function and convex envelope for these cases.
    """
    np.random.seed(seed)
    
    # Create the mixture distribution
    dist1 = UniformDistribution(0, 1)
    safe_epsilon = max(epsilon, 0.001)
    dist2_low = (1 / safe_epsilon) ** 2
    dist2_high = dist2_low + 1
    dist2 = UniformDistribution(dist2_low, dist2_high)

    if given_dist:
        dist = given_dist
    else:
        dist = MixtureDistribution(dist1, dist2, p1=(1-epsilon))
    
    # Generate sample
    # m = int(np.log(1 / delta) / (alpha ** 2)) + 100 # from paper
    m = num_samples
    print(f"Number of samples: {m}")
    samples = dist.sample(m)
    # Convert to list of single bidder samples format
    bidder_samples = [samples]
    
    # Print distribution info
    print(f"\nDebugging case: epsilon={epsilon}, alpha={alpha}")
    print(f"Mixture: U(0,1) with p={1-epsilon:.3f} and U({dist2_low:.2f},{dist2_high:.2f}) with p={epsilon:.3f}")
    
    # Run the tester and capture intermediate values
    n = 1  # Single bidder for debugging
    # m = num_samples
    delta = 0.05
    
    # Sort samples
    sorted_samples = np.sort(samples)
    
    # Compute empirical CDF
    empirical_cdf = np.arange(m) / m
    q_e_values = 1 - empirical_cdf
    
    
    # Compute quantile shifts
    c_values = []
    q_hat_values = []
    
    for j, v in enumerate(sorted_samples):
        if v < 0:
            continue
            
        q_e = q_e_values[j]
        c = np.sqrt(2 * q_e * (1 - q_e) * np.log(2*m*n/delta) / m) + \
            4 * np.log(2*m*n/delta) / m + alpha
        c_values.append(c)
        
        if v == 0:
            q_hat = 1
        else:
            q_hat = max(0, q_e - c)
        q_hat_values.append(q_hat)
    
    # Apply link function
    h_values = []
    for q_hat in q_hat_values:
        try:
            h_values.append(link_function(1 - q_hat))
        except:
            h_values.append(float('inf'))
    # print(f"Initial Link function values: {h_values}")
    
    # Compute convex envelope
    h_tilde_values = convex_envelope(sorted_samples[sorted_samples > 0], h_values)
    
    # Convert back to quantiles
    q_tilde_values = []
    for h_tilde in h_tilde_values:
        try:
            q_tilde_values.append(1 - inverse_link_function(h_tilde))
        except:
            q_tilde_values.append(float('inf'))
    
    # Check violations
    violations = []
    for j, (q_e, q_tilde, c) in enumerate(zip(q_e_values[sorted_samples > 0], 
                                             q_tilde_values, c_values)):
        diff = abs(q_tilde - q_e)
        if diff > (c + 0.00000001):
            violations.append((j, q_e, q_tilde, diff, c))
    
    # Now visualize everything
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    axs = axs.flatten()
    
    # Plot 1: Empirical CDF
    ax = axs[0]
    ax.plot(sorted_samples, empirical_cdf, 'b-', linewidth=2)
    ax.set_xlabel('Value (v)', fontsize=12)
    ax.set_ylabel('F(v)', fontsize=12)
    ax.set_title(f'Empirical CDF (epsilon={epsilon}, alpha={alpha})', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Quantile function
    ax = axs[1]
    ax.plot(sorted_samples, q_e_values, 'b-', linewidth=2, label='q_e (empirical)')
    ax.set_xlabel('Value (v)', fontsize=12)
    ax.set_ylabel('Quantile q(v)', fontsize=12)
    ax.set_title('Empirical Quantile Function', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 3: Link function before and after convex envelope
    ax = axs[2]
    valid_h = [h for h in h_values if h != float('inf')]
    if valid_h:
        ax.plot(sorted_samples[sorted_samples > 0], h_values, 'r.', alpha=0.5, label='h (link function)')
        ax.plot(sorted_samples[sorted_samples > 0], h_tilde_values, 'g-', linewidth=2, label='h_tilde (convex envelope)')
        # ax.set_ylim(0, np.percentile(valid_h, 95))  # Zoom in to see details
    ax.set_xlabel('Value (v)', fontsize=12)
    ax.set_ylabel('h(v)', fontsize=12)
    ax.set_title('Link Function and Convex Envelope', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    # Print the convex envelope values for debugging and their associated quantiles
    # print(f"\nConvex envelope values (h_tilde): {h_tilde_values}")
    
    # Plot 4: Quantile shifts and differences
    ax = axs[3]
    ax.plot(sorted_samples[sorted_samples > 0], q_e_values[sorted_samples > 0], 
           'b-', linewidth=2, label='q_e (empirical)')
    ax.plot(sorted_samples[sorted_samples > 0], q_hat_values, 
           'm-', linewidth=2, label='q_hat (shifted)')
    ax.plot(sorted_samples[sorted_samples > 0], q_tilde_values, 
           'g-', linewidth=2, label='q_tilde (after convex envelope)')
    ax.plot(sorted_samples[sorted_samples > 0], 
           [q - c for q, c in zip(q_e_values[sorted_samples > 0], c_values)], 
           'r--', linewidth=1, label='q_e - c (threshold)')
    ax.set_xlabel('Value (v)', fontsize=12)
    ax.set_ylabel('Quantile', fontsize=12)
    ax.set_title('Quantile Comparison', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 5: Quantile differences vs. shift values
    ax = axs[4]
    differences = [abs(q_tilde - q_e) for q_e, q_tilde in zip(q_e_values[sorted_samples > 0], q_tilde_values)]
    ax.plot(sorted_samples[sorted_samples > 0], differences, 'b-', linewidth=2, label='|q_tilde - q_e|')
    ax.plot(sorted_samples[sorted_samples > 0], c_values, 'r-', linewidth=2, label='c (shift threshold)')
    ax.set_xlabel('Value (v)', fontsize=12)
    ax.set_ylabel('Difference', fontsize=12)
    ax.set_title('Quantile Differences vs. Shift Threshold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 6: Violations
    ax = axs[5]
    if violations:
        violation_values = sorted_samples[sorted_samples > 0][[v[0] for v in violations]]
        violation_diffs = [v[3] for v in violations]
        violation_thresholds = [v[4] for v in violations]
        
        ax.scatter(violation_values, violation_diffs, c='r', s=50, label='Violations')
        ax.scatter(violation_values, violation_thresholds, c='g', s=30, marker='x', label='Thresholds')
        ax.set_xlabel('Value (v)', fontsize=12)
        ax.set_ylabel('|q_tilde - q_e| and c', fontsize=12)
        ax.set_title(f'Violations ({len(violations)} points)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No Violations Detected', ha='center', va='center', fontsize=16)
        ax.set_title('Violations', fontsize=14)
    
    plt.tight_layout()
    # plt.savefig(f'plots/debugging/debug_epsilon_{epsilon}_alpha_{alpha}.png', dpi=300)
    plt.savefig(f'plots/debugging/gauss/debug_epsilon_{epsilon}_alpha_{alpha}.png', dpi=300)
    # plt.show()
    
    # Run the actual tester to confirm result
    tester_result = regularized_tester(bidder_samples, alpha, delta)
    print(f"\nTester result: {tester_result}")
    print(f"Number of violations: {len(violations)}")
    if violations:
        print(f"First few violations:")
        for v in violations[:5]:
            print(f"  At index {v[0]}: value={sorted_samples[sorted_samples > 0][v[0]]:.4f}, "
                  f"diff={v[3]:.6f}, threshold={v[4]:.6f}")
    
    return tester_result, violations, sorted_samples, q_e_values, q_tilde_values, h_values, h_tilde_values


# function to compare multiple problematic cases
def debug_multiple_cases(epsilon_values=[0, 0.01, 0.05, 0.1, 0.2, 0.3], alpha_values=[0.01, 0.05, 0.1, 0.2, 0.5], num_samples=1000):
    """
    Debug multiple cases to understand the pattern of unexpected behavior.
    """
    results = []
    
    for epsilon in epsilon_values:
        for alpha in alpha_values:
            print(f"\n{'='*50}")
            print(f"Testing epsilon={epsilon}, alpha={alpha}")
            print(f"{'='*50}")
            
            result = debug_specific_case(epsilon=epsilon, alpha=alpha, num_samples=num_samples)
            results.append({
                'epsilon': epsilon,
                'alpha': alpha,
                'tester_result': result[0],
                'epsilon/alpha ratio': epsilon / alpha,
                'num_violations': len(result[1]),
                'first_violation': result[1][0] if result[1] else None
            })
    
    # Create a summary visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Create a matrix for visualization
    matrix = np.zeros((len(epsilon_values), len(alpha_values)))
    for i, epsilon in enumerate(epsilon_values):
        for j, alpha in enumerate(alpha_values):
            result = next(r for r in results if r['epsilon'] == epsilon and r['alpha'] == alpha)
            matrix[i, j] = 1 if result['tester_result'] == "YES" else 0
    
    im = ax.imshow(matrix, cmap='viridis', aspect='auto')
    ax.set_xticks(np.arange(len(alpha_values)))
    ax.set_yticks(np.arange(len(epsilon_values)))
    ax.set_xticklabels([f'α={a}' for a in alpha_values])
    ax.set_yticklabels([f'ε={e}' for e in epsilon_values])
    
    # Add text annotations
    for i in range(len(epsilon_values)):
        for j in range(len(alpha_values)):
            result = next(r for r in results if r['epsilon'] == epsilon_values[i] and r['alpha'] == alpha_values[j])
            text = f"{result['tester_result']}\n{result['epsilon/alpha ratio']:.1f} Epsilon/Alpha."
            ax.text(j, i, text, ha="center", va="center", color="w")
    
    ax.set_title('Tester Results and Violations')
    plt.colorbar(im, ax=ax, ticks=[0, 1], label='Tester Result (0=NO, 1=YES)')
    plt.tight_layout()
    plt.savefig('plots/debugging/debug_summary.png', dpi=300)
    plt.show()
    
    return results

def plot_tester_decisions_vs_epsilon_alpha_ratio(epsilon_values=[0.01, 0.02, 0.05, 0.1, 0.2, 0.3], 
                                                alpha_values=[0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5],
                                                sample_size=1000):
    """
    Plot the tester's decisions (YES/NO) against the ratio of epsilon/alpha.
    
    Parameters:
    - epsilon_values: List of epsilon values to test
    - alpha_values: List of alpha values to test
    - sample_size: Number of samples to use for each test
    """
    # Create lists to store results
    ratios = []
    decisions = []
    all_epsilons = []
    all_alphas = []

    np.random.seed(22)  # For reproducibility
    
    # Run tests for all combinations of epsilon and alpha
    for epsilon in epsilon_values:
        # Create the distribution
        dist1 = UniformDistribution(0, 1)
        safe_epsilon = max(epsilon, 0.001)  # Avoid division by zero
        dist1_low = 2 ** (1 / safe_epsilon)
        dist1_high = dist1_low + 1
        dist2 = UniformDistribution(dist1_low, dist1_high)
        dist = MixtureDistribution(dist1, dist2, p1=(1-epsilon))
        
        # Generate samples
        # samples = [dist.sample(sample_size)]
        
        for alpha in alpha_values:
            delta = 0.05  # Confidence parameter
            # m = int(np.log(1 / delta) / (alpha ** 2)) + 100 # from paper + buffer
            m = sample_size
            samples = [dist.sample(m)]
        
            # Run tester
            tester_result = regularized_tester(samples, alpha)
            
            # Store results
            ratio = epsilon / alpha
            ratios.append(ratio)
            decisions.append(1 if tester_result == "YES" else 0)
            all_epsilons.append(epsilon)
            all_alphas.append(alpha)
            
            print(f"Epsilon: {epsilon:.4f}, Alpha: {alpha:.4f}, Ratio: {ratio:.4f}, Result: {tester_result}")
    
    # Create scatter plot
    plt.figure(figsize=(12, 8))
    
    # Size points based on alpha for better visualization (smaller alpha = larger point)
    sizes = [300 / (a * 100) for a in all_alphas]
    
    # Scatter plot with different colors for YES/NO
    for decision in [0, 1]:
        indices = [i for i, d in enumerate(decisions) if d == decision]
        plt.scatter([ratios[i] for i in indices], 
                    [decisions[i] for i in indices],
                    c=['red' if decision == 0 else 'green'],
                    s=[sizes[i] for i in indices],
                    alpha=0.6,
                    label='NO' if decision == 0 else 'YES')
    
    # Add annotations showing epsilon and alpha values
    for i, (ratio, decision) in enumerate(zip(ratios, decisions)):
        plt.annotate(f"ε={all_epsilons[i]:.3f}\nα={all_alphas[i]:.3f}", 
                    (ratio, decision),
                    textcoords="offset points",
                    xytext=(0, 10 if decision == 1 else -20),
                    ha='center',
                    fontsize=8)
    
    # Add a vertical line at ratio = 1
    plt.axvline(x=1, color='gray', linestyle='--', alpha=0.7, label='ε = α')
    
    # Set up the plot
    plt.title('Tester Decision Heatmap (Epsilon vs Alpha)')
    plt.xlabel('Epsilon/Alpha Ratio (log scale)')
    plt.ylabel('Tester Decision (0=NO, 1=YES)')
    plt.ylim(-0.5, 1.5)  # Add some padding
    plt.yticks([0, 1], ['NO', 'YES'])
    plt.xscale('log')  # Log scale for better visualization
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Create a second version with jittered y values for better visualization
    plt.figure(figsize=(12, 8))
    
    # Create jittered y values
    jittered_decisions = [d + np.random.uniform(-0.1, 0.1) for d in decisions]
    
    # Create scatter plot with jittered values
    for decision in [0, 1]:
        indices = [i for i, d in enumerate(decisions) if d == decision]
        plt.scatter([ratios[i] for i in indices], 
                    [jittered_decisions[i] for i in indices],
                    c=['red' if decision == 0 else 'green'],
                    s=[sizes[i] for i in indices],
                    alpha=0.6,
                    label='NO' if decision == 0 else 'YES')
    
    # Add annotations
    for i, (ratio, jittered) in enumerate(zip(ratios, jittered_decisions)):
        plt.annotate(f"ε={all_epsilons[i]:.3f}\nα={all_alphas[i]:.3f}", 
                    (ratio, jittered),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha='center',
                    fontsize=8)
    
    plt.axvline(x=1, color='gray', linestyle='--', alpha=0.7, label='ε = α')
    
    # Add a threshold line
    plt.axhline(y=0.5, color='black', linestyle=':', alpha=0.5)
    
    plt.title('Tester Decision vs Epsilon/Alpha Ratio (Jittered for Visibility)')
    plt.xlabel('Epsilon/Alpha Ratio (log scale)')
    plt.ylabel('Tester Decision (Jittered for Visibility)')
    plt.yticks([0, 1], ['NO', 'YES'])
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save and show
    plt.savefig('tester_decision_vs_ratio.png', dpi=300)
    plt.show()
    
    # Create heatmap version
    plt.figure(figsize=(12, 8))
    
    # Create matrix for heatmap
    matrix = np.zeros((len(epsilon_values), len(alpha_values)))
    for i, epsilon in enumerate(epsilon_values):
        for j, alpha in enumerate(alpha_values):
            # Find the corresponding decision
            for idx, (eps, alp, dec) in enumerate(zip(all_epsilons, all_alphas, decisions)):
                if eps == epsilon and alp == alpha:
                    matrix[i, j] = dec
                    break
    
    # Create heatmap
    im = plt.imshow(matrix, cmap='RdYlGn', aspect='auto', alpha=0.8)
    
    # Add text annotations
    for i in range(len(epsilon_values)):
        for j in range(len(alpha_values)):
            # Calculate ratio for the annotation
            ratio = epsilon_values[i] / alpha_values[j]
            plt.text(j, i, f"{ratio:.2f}\n{'YES' if matrix[i, j] == 1 else 'NO'}", 
                    ha="center", va="center", color="black", fontsize=9)
    
    # Set tick labels
    plt.yticks(np.arange(len(epsilon_values)), [f"ε={e:.3f}" for e in epsilon_values])
    plt.xticks(np.arange(len(alpha_values)), [f"α={a:.3f}" for a in alpha_values])
    
    plt.colorbar(im, ticks=[0, 1], label='Tester Decision')
    plt.title('Tester Decision Heatmap (Epsilon vs Alpha)')
    plt.xlabel('Alpha Value')
    plt.ylabel('Epsilon Value')
    plt.tight_layout()
    
    # Save and show
    plt.savefig('tester_decision_heatmap.png', dpi=300)
    plt.show()
    
    # Return the raw data for further analysis
    return {
        'ratios': ratios,
        'decisions': decisions,
        'epsilons': all_epsilons,
        'alphas': all_alphas
    }

def test_gaussian_mixture_single(alpha=0.01, epsilon=0.1, delta=0.05, p=0.9):
    dist1 = GaussianDistribution(1, 1)
    dist2_mean = (1 / epsilon) ** 2
    dist2 = GaussianDistribution(dist2_mean, 1)
    dist = MixtureDistribution(dist1, dist2, p1=(1-epsilon))
    # m = int(np.log(1 / delta) / (alpha ** 2)) + 100 # from paper
    m = 100000
    # samples = dist.sample(m)
    
    # show pdf for debugging
    # plt.hist(samples, bins=100, density=True, alpha=0.5)
    # plt.title('Sample Distribution')
    # plt.xlabel('Value')
    # plt.show()

    return debug_specific_case(epsilon=epsilon, alpha=alpha, num_samples=m, given_dist=dist)


def test_gaussian_mixture_multiple(epsilon_values=[0.01, 0.05, 0.1, 0.2, 0.3], alpha_values=[0.01, 0.05, 0.1, 0.2, 0.5], num_samples=1000):
    """
    Debug multiple cases to understand the pattern of unexpected behavior.
    """
    results = []
    
    for epsilon in epsilon_values:
        for alpha in alpha_values:
            print(f"\n{'='*50}")
            print(f"Testing epsilon={epsilon}, alpha={alpha}")
            print(f"{'='*50}")
            
            result = test_gaussian_mixture_single(epsilon=epsilon, alpha=alpha)
            results.append({
                'epsilon': epsilon,
                'alpha': alpha,
                'tester_result': result[0],
                'epsilon/alpha ratio': epsilon / alpha,
                'num_violations': len(result[1]),
                'first_violation': result[1][0] if result[1] else None
            })
    
    # Create a summary visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Create a matrix for visualization
    matrix = np.zeros((len(epsilon_values), len(alpha_values)))
    for i, epsilon in enumerate(epsilon_values):
        for j, alpha in enumerate(alpha_values):
            result = next(r for r in results if r['epsilon'] == epsilon and r['alpha'] == alpha)
            matrix[i, j] = 1 if result['tester_result'] == "YES" else 0
    
    im = ax.imshow(matrix, cmap='viridis', aspect='auto')
    ax.set_xticks(np.arange(len(alpha_values)))
    ax.set_yticks(np.arange(len(epsilon_values)))
    ax.set_xticklabels([f'α={a}' for a in alpha_values])
    ax.set_yticklabels([f'ε={e}' for e in epsilon_values])
    ax.set_xlabel('Alpha Value')
    ax.set_ylabel('Epsilon Value')
    
    # Add text annotations
    for i in range(len(epsilon_values)):
        for j in range(len(alpha_values)):
            result = next(r for r in results if r['epsilon'] == epsilon_values[i] and r['alpha'] == alpha_values[j])
            text = f"{result['epsilon/alpha ratio']:.1f}\n{result['tester_result']}"
            # ax.text(j, i, text, ha="center", va="center", color="w")
            if result['tester_result'] == "YES":
                ax.text(j, i, text, ha="center", va="center", color="black")
            else:
                ax.text(j, i, text, ha="center", va="center", color="white")
    
    ax.set_title('Tester Decision Heatmap (Epsilon vs Alpha)')
    plt.colorbar(im, ax=ax, ticks=[0, 1], label='Tester Result (0=NO, 1=YES)')
    plt.tight_layout()
    plt.savefig('plots/debugging/gauss/debug_summary.png', dpi=300)
    plt.show()
    
    return results

def test_mixed_distributions(epsilon_values=[0.01, 0.05, 0.1, 0.2, 0.3], alpha_values=[0.01, 0.05, 0.1, 0.2, 0.5], num_samples=1000, dist_type='None'):
    """
    Debug multiple cases to understand the pattern of unexpected behavior.
    """
    results = []
    
    for epsilon in epsilon_values:
        for alpha in alpha_values:
            print(f"\n{'='*50}")
            print(f"Testing epsilon={epsilon}, alpha={alpha}")
            print(f"{'='*50}")
            
            result = test_gaussian_mixture_single(epsilon=epsilon, alpha=alpha)
            if distribution_type == 'Gaussian':
                test_gaussian_mixture_single(epsilon=epsilon, alpha=alpha)
            elif distribution_type == 'LogNormal':
                test_lognormal_mixture_single(epsilon=epsilon, alpha=alpha)
            elif distribution_type == 'Uniform':
                test_uniform_mixture_single(epsilon=epsilon, alpha=alpha)
            else:
                raise ValueError("Invalid distribution type. Choose 'Gaussian', 'LogNormal', or 'Uniform'.")
            results.append({
                'epsilon': epsilon,
                'alpha': alpha,
                'tester_result': result[0],
                'epsilon/alpha ratio': epsilon / alpha,
                'num_violations': len(result[1]),
                'first_violation': result[1][0] if result[1] else None
            })
    
    # Create a summary visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Create a matrix for visualization
    matrix = np.zeros((len(epsilon_values), len(alpha_values)))
    for i, epsilon in enumerate(epsilon_values):
        for j, alpha in enumerate(alpha_values):
            result = next(r for r in results if r['epsilon'] == epsilon and r['alpha'] == alpha)
            matrix[i, j] = 1 if result['tester_result'] == "YES" else 0
    
    im = ax.imshow(matrix, cmap='viridis', aspect='auto')
    ax.set_xticks(np.arange(len(alpha_values)))
    ax.set_yticks(np.arange(len(epsilon_values)))
    ax.set_xticklabels([f'α={a}' for a in alpha_values])
    ax.set_yticklabels([f'ε={e}' for e in epsilon_values])
    ax.set_xlabel('Alpha Value')
    ax.set_ylabel('Epsilon Value')
    
    # Add text annotations
    for i in range(len(epsilon_values)):
        for j in range(len(alpha_values)):
            result = next(r for r in results if r['epsilon'] == epsilon_values[i] and r['alpha'] == alpha_values[j])
            text = f"{result['epsilon/alpha ratio']:.1f}\n{result['tester_result']}"
            # ax.text(j, i, text, ha="center", va="center", color="w")
            if result['tester_result'] == "YES":
                ax.text(j, i, text, ha="center", va="center", color="black")
            else:
                ax.text(j, i, text, ha="center", va="center", color="white")
    
    ax.set_title('Tester Decision Heatmap (Epsilon vs Alpha)')
    plt.colorbar(im, ax=ax, ticks=[0, 1], label='Tester Result (0=NO, 1=YES)')
    plt.tight_layout()
    plt.savefig('plots/debugging/gauss/debug_summary.png', dpi=300)
    plt.show()
    
    return results




if __name__ == "__main__":
    # Uncomment the following line to debug a specific case
    # result = debug_specific_case(epsilon=0.3, alpha=0.2)
    
    # Uncomment the following line to debug multiple cases
    # results = debug_multiple_cases()
# 
    # Uncomment the following line to plot tester decisions vs epsilon/alpha ratio
    # plot_tester_decisions_vs_epsilon_alpha_ratio(sample_size=1000000)

    # test_gaussian_mixture_single()
    test_gaussian_mixture_multiple()