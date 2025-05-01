import numpy as np
import matplotlib.pyplot as plt
from src.distributions import UniformDistribution, ProductDistribution, MixtureDistribution, GaussianDistribution, LogNormalDistribution
from src.algorithms import regularized_tester, regularized_learner
from src.utils import link_function, inverse_link_function, convex_envelope

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





if __name__ == "__main__":
    # Uncomment the following line to plot tester decisions vs epsilon/alpha ratio
    plot_tester_decisions_vs_epsilon_alpha_ratio(sample_size=1000000)
