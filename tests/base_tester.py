import numpy as np
import matplotlib.pyplot as plt
from src.algorithms import regularized_tester, regularized_learner
from src.utils import link_function, inverse_link_function, convex_envelope

class BaseTester:
    """Base class for testing regularized learning algorithms with different distributions."""
    
    def __init__(self, seed=22):
        """Initialize the tester with a seed for reproducibility."""
        self.seed = seed
        np.random.seed(seed)
        
    def test_single_case(self, distribution, epsilon=0.1, alpha=0.01, num_samples=1000, delta=0.05):
        """
        Test a single case with the given distribution and parameters.
        
        Args:
            distribution: The distribution to test
            epsilon: The corruption parameter
            alpha: The distortion parameter
            num_samples: Number of samples to use
            delta: Confidence parameter
            
        Returns:
            tuple: (tester_result, violations, sorted_samples, q_e_values, 
                   q_tilde_values, h_values, h_tilde_values)
        """
        print(f"\nTesting case: epsilon={epsilon}, alpha={alpha}")
        print(f"Distribution: {distribution}")
        
        # Generate samples
        m = num_samples
        samples = distribution.sample(m)
        bidder_samples = [samples]  # Format for the tester
        
        # Run the analysis
        result_data = self._analyze_samples(bidder_samples, samples, epsilon, alpha, delta)
        
        # Visualize the results
        self._visualize_results(result_data, epsilon, alpha)
        
        # Run the actual tester
        tester_result = regularized_tester(bidder_samples, alpha, delta)
        print(f"\nTester result: {tester_result}")
        print(f"Number of violations: {len(result_data['violations'])}")
        
        if result_data['violations']:
            print(f"First few violations:")
            sorted_samples = result_data['sorted_samples']
            for v in result_data['violations'][:5]:
                print(f"  At index {v[0]}: value={sorted_samples[sorted_samples > 0][v[0]]:.4f}, "
                      f"diff={v[3]:.6f}, threshold={v[4]:.6f}")
        
        return tester_result, result_data
    
    def test_multiple_cases(self, get_distribution_fn, epsilon_values, alpha_values, num_samples=1000):
        """
        Test multiple cases to understand patterns of behavior.
        
        Args:
            get_distribution_fn: Function that returns a distribution given epsilon
            epsilon_values: List of epsilon values to test
            alpha_values: List of alpha values to test
            num_samples: Number of samples for each test
            
        Returns:
            list: Results for all test cases
        """
        results = []
        
        for epsilon in epsilon_values:
            distribution = get_distribution_fn(epsilon)
            
            for alpha in alpha_values:
                print(f"\n{'='*50}")
                print(f"Testing epsilon={epsilon}, alpha={alpha}")
                print(f"{'='*50}")
                
                tester_result, result_data = self.test_single_case(
                    distribution, epsilon, alpha, num_samples
                )
                
                results.append({
                    'epsilon': epsilon,
                    'alpha': alpha,
                    'tester_result': tester_result,
                    'epsilon/alpha ratio': epsilon / alpha,
                    'num_violations': len(result_data['violations']),
                    'first_violation': result_data['violations'][0] if result_data['violations'] else None
                })
        
        # Create summary visualization
        self._visualize_summary(results, epsilon_values, alpha_values)
        
        return results
    
    def _analyze_samples(self, bidder_samples, samples, epsilon, alpha, delta):
        """Analyze samples and compute intermediate values for visualization."""
        n = 1  # Single bidder for debugging
        m = len(samples)
        
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
        
        return {
            'sorted_samples': sorted_samples,
            'empirical_cdf': empirical_cdf,
            'q_e_values': q_e_values,
            'c_values': c_values,
            'q_hat_values': q_hat_values,
            'h_values': h_values,
            'h_tilde_values': h_tilde_values,
            'q_tilde_values': q_tilde_values,
            'violations': violations
        }
    
    def _visualize_results(self, data, epsilon, alpha):
        """Visualize results of a single test case."""
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))
        axs = axs.flatten()
        
        # Plot 1: Empirical CDF
        ax = axs[0]
        ax.plot(data['sorted_samples'], data['empirical_cdf'], 'b-', linewidth=2)
        ax.set_xlabel('Value (v)', fontsize=12)
        ax.set_ylabel('F(v)', fontsize=12)
        ax.set_title(f'Empirical CDF (epsilon={epsilon}, alpha={alpha})', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Quantile function
        ax = axs[1]
        ax.plot(data['sorted_samples'], data['q_e_values'], 'b-', linewidth=2, label='q_e (empirical)')
        ax.set_xlabel('Value (v)', fontsize=12)
        ax.set_ylabel('Quantile q(v)', fontsize=12)
        ax.set_title('Empirical Quantile Function', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 3: Link function before and after convex envelope
        ax = axs[2]
        valid_h = [h for h in data['h_values'] if h != float('inf')]
        if valid_h:
            ax.plot(data['sorted_samples'][data['sorted_samples'] > 0], data['h_values'], 
                   'r.', alpha=0.5, label='h (link function)')
            ax.plot(data['sorted_samples'][data['sorted_samples'] > 0], data['h_tilde_values'], 
                   'g-', linewidth=2, label='h_tilde (convex envelope)')
        ax.set_xlabel('Value (v)', fontsize=12)
        ax.set_ylabel('h(v)', fontsize=12)
        ax.set_title('Link Function and Convex Envelope', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 4: Quantile shifts and differences
        ax = axs[3]
        ax.plot(data['sorted_samples'][data['sorted_samples'] > 0], 
               data['q_e_values'][data['sorted_samples'] > 0], 
               'b-', linewidth=2, label='q_e (empirical)')
        ax.plot(data['sorted_samples'][data['sorted_samples'] > 0], data['q_hat_values'], 
               'm-', linewidth=2, label='q_hat (shifted)')
        ax.plot(data['sorted_samples'][data['sorted_samples'] > 0], data['q_tilde_values'], 
               'g-', linewidth=2, label='q_tilde (after convex envelope)')
        ax.plot(data['sorted_samples'][data['sorted_samples'] > 0], 
               [q - c for q, c in zip(data['q_e_values'][data['sorted_samples'] > 0], data['c_values'])], 
               'r--', linewidth=1, label='q_e - c (threshold)')
        ax.set_xlabel('Value (v)', fontsize=12)
        ax.set_ylabel('Quantile', fontsize=12)
        ax.set_title('Quantile Comparison', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 5: Quantile differences vs. shift values
        ax = axs[4]
        differences = [abs(q_tilde - q_e) for q_e, q_tilde in 
                      zip(data['q_e_values'][data['sorted_samples'] > 0], data['q_tilde_values'])]
        ax.plot(data['sorted_samples'][data['sorted_samples'] > 0], differences, 
               'b-', linewidth=2, label='|q_tilde - q_e|')
        ax.plot(data['sorted_samples'][data['sorted_samples'] > 0], data['c_values'], 
               'r-', linewidth=2, label='c (shift threshold)')
        ax.set_xlabel('Value (v)', fontsize=12)
        ax.set_ylabel('Difference', fontsize=12)
        ax.set_title('Quantile Differences vs. Shift Threshold', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 6: Violations
        ax = axs[5]
        if data['violations']:
            violation_indices = [v[0] for v in data['violations']]
            violation_values = data['sorted_samples'][data['sorted_samples'] > 0][violation_indices]
            violation_diffs = [v[3] for v in data['violations']]
            violation_thresholds = [v[4] for v in data['violations']]
            
            ax.scatter(violation_values, violation_diffs, c='r', s=50, label='Violations')
            ax.scatter(violation_values, violation_thresholds, c='g', s=30, marker='x', label='Thresholds')
            ax.set_xlabel('Value (v)', fontsize=12)
            ax.set_ylabel('|q_tilde - q_e| and c', fontsize=12)
            ax.set_title(f'Violations ({len(data["violations"])} points)', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No Violations Detected', ha='center', va='center', fontsize=16)
            ax.set_title('Violations', fontsize=14)
        
        plt.tight_layout()
        dist_type = getattr(self, 'dist_type', 'general')
        plt.savefig(f'plots/debugging/{dist_type}/debug_epsilon_{epsilon}_alpha_{alpha}.png', dpi=300)
        plt.close()
    
    def _visualize_summary(self, results, epsilon_values, alpha_values):
        """Visualize summary of multiple test cases."""
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
                if result['tester_result'] == "YES":
                    ax.text(j, i, text, ha="center", va="center", color="black")
                else:
                    ax.text(j, i, text, ha="center", va="center", color="white")
        
        dist_type = getattr(self, 'dist_type', 'general')
        ax.set_title(f'{dist_type} Distribution: Tester Decision Heatmap')
        plt.colorbar(im, ax=ax, ticks=[0, 1], label='Tester Result (0=NO, 1=YES)')
        plt.tight_layout()
        plt.savefig(f'plots/debugging/{dist_type}/debug_summary.png', dpi=300)
        plt.close()