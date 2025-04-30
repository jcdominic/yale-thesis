import numpy as np
from src.utils import link_function, inverse_link_function, convex_envelope, empirical_cdf
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def regularized_tester(samples, alpha, delta=0.05):
    """
    Implementation of Algorithm 1: Regularized Tester

    Parameters:
    - samples: list of samples from each bidder [samples_1, samples_2, ..., samples_n]
    - alpha: distortion parameter
    - delta: confidence parameter
    
    Returns:
    - "YES" if there exists a regular distribution close to the empirical distribution
    - "NO" otherwise
    """
    n = len(samples)  # Number of bidders
    m = len(samples[0])  # Number of samples per bidder
    # print(f"Number of bidders: {n}, Number of samples per bidder: {m}")
    # print(f"First sample: {samples[0][-1:-5:-1]}...")  # Print 5 samples for debugging
    
    for i in range(n):
        # Sort samples for this bidder
        bidder_samples = np.sort(samples[i])
        # Print few samples for debugging
        # print(f"Bidder {i} samples: {bidder_samples[-1:-5:-1]}...")  # Print first 5 samples for debugging
        
        # Compute empirical CDF
        # empirical_cdf = np.arange(1, m+1) / m
        # print min and max of samples
        # print(f"Bidder {i} min: {min(bidder_samples):.6f}, max: {max(bidder_samples):.6f}") # looks good

        empirical_cdf = np.arange(m) / m
        # np.sort(empirical_cdf)
        # print(f"Domain of Empirical CDF: {bidder_samples[0]:.6f} to {bidder_samples[-1]:.6f}")
    

        # # Plot empirical CDF for debugging. If output put it in the plots folder
        # plt.plot(bidder_samples, empirical_cdf, label=f'Bidder {i}')
        # plt.title("Empirical CDF")
        # plt.xlabel("Value (v)")
        # plt.ylabel("F_n(v)")
        # plt.legend()
        # plt.savefig(f'plots/empirical_cdf_bidder_{i}.png')

  
        
        # Compute empirical quantiles
        q_e_values = 1 - empirical_cdf
        
        # Compute the quantile shifts for all points
        c_values = []
        q_hat_values = []
        
        for j, v in enumerate(bidder_samples):
            if v < 0:
                continue
                
            q_e = q_e_values[j]
            
            # Compute the quantile shift
            c = np.sqrt(2 * q_e * (1 - q_e) * np.log(2*m*n/delta) / m) + \
                4 * np.log(2*m*n/delta) / m + alpha
            c_values.append(c)
            
            # Calculate q_hat
            if v == 0:
                q_hat = 1
            else:
                q_hat = max(0, q_e - c)
            q_hat_values.append(q_hat)
            # q_hat = max(0, q_e - c)
            # q_hat_values.append(q_hat)
        
        # Apply link function to all points 
        # Pg 4 -- Conv(D) denotes the convex envelope of the distribution's link function
        h_values = []
        # failure_count = 0
        for q_hat in q_hat_values:
            try:
                h_values.append(link_function(1 - q_hat))
            except:
                # Handle division by zero or other numerical issues
                # print(f"Warning: link_function failed for q_hat={q_hat}, setting h to infinity")
                # failure_count += 1
                h_values.append(float('inf'))
                # h_values.append(link_function(1 - 0.01))  # Set to a default value (e.g., link_function(1))

        # print(f"Number of failures in link function: {failure_count}")
        
        # Compute convex envelope of all points together
        h_tilde_values = convex_envelope(bidder_samples[bidder_samples > 0], h_values)

        # TODO: For the weird examples, plot the link function after the convex envelope
        
        # Convert back to quantiles
        q_tilde_values = []
        for h_tilde in h_tilde_values:
            try:
                q_tilde_values.append(1 - inverse_link_function(h_tilde))
            except:
                q_tilde_values.append(float('inf'))
        
        # Check if all quantiles are within allowed shifts
        for j, (q_e, q_tilde, c) in enumerate(zip(q_e_values[bidder_samples > 0], 
                                                 q_tilde_values, c_values)):
            diff = abs(q_tilde - q_e)
            # diff_count = 0
            if diff > (c + 0.00000001): # adding very small bufer for floating point errors ARGH
            # if diff > c:
                # diff_count += 1
                # print(f"Violation at index {j}: q_e={q_e}, q_tilde={q_tilde}, diff={diff}, c={c}")
                return "NO"
                # continue
    
    return "YES"


def regularized_learner(samples, alpha, delta=0.05):
    """
    Implementation of Algorithm 2: Regularized Learner

    Parameters:
    - samples: list of samples from each bidder [samples_1, samples_2, ..., samples_n]
    - alpha: distortion parameter
    - delta: confidence parameter
    
    Returns:
    - Reserve prices for each bidder
    """
    n = len(samples)  # Number of bidders
    m = len(samples[0])  # Number of samples per bidder
    reserve_prices = []
    
    for i in range(n):
        # Sort samples for this bidder
        bidder_samples = np.sort(samples[i])
        positive_indices = bidder_samples > 0
        positive_samples = bidder_samples[positive_indices]
        
        if len(positive_samples) == 0:
            reserve_prices.append(0)  # No positive samples
            continue
            
        # Compute empirical CDF
        v, empirical_cdf_vals = empirical_cdf(positive_samples)
        
        # Compute empirical quantiles
        q_e_values = 1 - empirical_cdf_vals
        
        # Compute the quantile shifts
        c_values = []
        q_hat_values = []
        
        for j, q_e in enumerate(q_e_values):
            # Compute the quantile shift
            c = np.sqrt(2 * q_e * (1 - q_e) * np.log(2*m*n/delta) / m) + \
                4 * np.log(2*m*n/delta) / m + alpha
            c_values.append(c)
            
            # Calculate q_hat
            q_hat = max(0, q_e - c)
            q_hat_values.append(q_hat)
        
        # Apply link function
        h_values = []
        for q_hat in q_hat_values:
            try:
                h_values.append(link_function(1 - q_hat))
            except:
                h_values.append(float('inf'))
        
        # Take convex envelope of link function
        h_tilde_values = convex_envelope(positive_samples, h_values)
        
        # Convert back to quantiles
        q_tilde_values = []
        for h_tilde in h_tilde_values:
            try:
                q_tilde_values.append(1 - inverse_link_function(h_tilde))
            except:
                q_tilde_values.append(float('inf'))
        
        # Construct E' by further shifting quantiles
        q_prime_values = [max(0, q - alpha) for q in q_tilde_values]

        # Maximize revenue for the single bidder case
        revenue_values = bidder_samples * np.array(q_prime_values)
        
        # Find the value that maximizes revenue
        max_revenue_idx = np.argmax(revenue_values)
        reserve_price = bidder_samples[max_revenue_idx]
        reserve_prices.append(reserve_price)
    
    
    return reserve_prices


import numpy as np
import matplotlib.pyplot as plt

def debug_regularized_tester_once(alpha=0.01, delta=0.05, m=500, seed=22):
    """
    Generate uniform(0,1) samples for ONE bidder, run regularized_tester,
    and visually plot the empirical CDF, shifts, and resulting quantiles.
    """
    np.random.seed(seed)
    
    # -----------------------
    # 1) Generate samples
    # -----------------------
    # samples = np.random.rand(m)  # uniform(0,1)
    # epsilon = 1e-10  # A small number to "nudge" away from 0 and 1
    # samples = epsilon + (1 - 2 * epsilon) * np.random.rand(m)
    # samples.sort()
    # # Print min and max for debugging
    # print(f"Samples min: {samples[0]:.6f}, max: {samples[-1]:.6f}")

    from src.distributions import MixtureDistribution
    from src.distributions import UniformDistribution

    # # Create the irregular distribution (a mixture)
    dist1 = UniformDistribution(0, 1)
    dist2 = UniformDistribution(10, 11)
    
    dist = MixtureDistribution(dist1, dist2, p1=0.9)
    # dist = dist1
    
    # Generate samples  
    num_samples = 100
    samples = dist.sample(num_samples)
    # print(samples)
    # samples[0].sort()
    samples.sort()

    n = len(samples)  # Number of bidders
    # m = len(samples[0])  # Number of samples per bidder
    m = n
    # print(f"Number of bidders: {n}, Number of samples per
    
    # We'll store intermediate values so we can visualize them
    c_list = []
    q_e_list = []
    q_hat_list = []
    h_list = []
    h_tilde_list = []
    q_tilde_list = []
    
    # -----------------------
    # 2) Empirical CDF
    # -----------------------
    # Option A 
    #   empirical_cdf = np.arange(1, m + 1) / m
    #
    # Option B 
    #   empirical_cdf = np.arange(m) / m
    #
    # For better interpretability, we'll do Option A here:
    empirical_cdf = np.arange(m) / m

    # Empirical quantiles: q_e = 1 - F(v)
    q_e_values = 1 - empirical_cdf
    
    # -----------------------
    # 3) Compute shifts c, then q_hat
    # -----------------------
    n = 1  # only 1 bidder in this debug scenario
    for j, v in enumerate(samples):
        # (Skip negative or zero values if your code does)
        # For uniform(0,1), all are > 0, so no skip
        q_e = q_e_values[j]
        
        # c = sqrt(...) + ... + alpha
        c = (
            np.sqrt(2 * q_e * (1 - q_e) * np.log(2 * m * n / delta) / m)
            + 4 * np.log(2 * m * n / delta) / m
            + alpha
        )
        
        # if v == 0:
        #     q_hat = 1
        # else:
        #     q_hat = max(0, q_e - c)
        q_hat = max(0, q_e - c)
        
        c_list.append(c)
        q_e_list.append(q_e)
        q_hat_list.append(q_hat)
    
    # -----------------------
    # 4) Apply link function
    # -----------------------
    from src.utils import link_function, inverse_link_function, convex_envelope
    
    for q_hat in q_hat_list:
        try:
            val = link_function(1 - q_hat)
        except:
            val = float('inf')
        h_list.append(val)
    
    # -----------------------
    # 5) Convex envelope
    # -----------------------
    # The code uses only positive samples, but let's feed in
    # all sorted > 0 samples (all of them in uniform(0,1)):
    h_tilde_values = convex_envelope(samples, h_list)
    
    # Convert h_tilde back to quantiles q_tilde
    for h_tilde in h_tilde_values:
        try:
            val = 1 - inverse_link_function(h_tilde)
        except:
            val = 0
            print("Warning: inverse_link_function failed, setting q_tilde to 0")
        q_tilde_list.append(val)
    
    # -----------------------
    # 6) Plot for debugging
    # -----------------------
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()
    
    # a) Empirical CDF
    axs[0].plot(samples, empirical_cdf, 'b.-', label='Empirical CDF')
    axs[0].set_title("Empirical CDF vs. Samples")
    axs[0].set_xlabel("Value (v)")
    axs[0].set_ylabel("F_n(v)")
    axs[0].legend()
    axs[0].grid(True)

    # Empirical CDF vs Quantiles over samples
    # axs[0].plot(samples, empirical_cdf, 'b.-', label='Empirical CDF')
    # axs[0].plot(samples, q_e_values, 'r.-', label='Empirical Quantiles')
    # axs[0].set_title("Empirical CDF vs. Quantiles")
    # axs[0].set_xlabel("Value (v)")
    # axs[0].set_ylabel("F_n(v) / q_e")
    # axs[0].legend()
    # axs[0].grid(True)
   
    
    # b) q_e vs. index + compare with q_hat, q_tilde
    idx = np.arange(m)
    axs[1].plot(idx, q_e_list, 'b.-', label='q_e (empirical)')
    axs[1].plot(idx, q_hat_list, 'g.-', label='q_hat (after shift)')
    axs[1].plot(idx, q_tilde_list, 'r.-', label='q_tilde (after convex env)')
    # axs[1].plot(idx, empirical_cdf, 'k--', label='Empirical CDF')
    axs[1].set_title("Quantiles vs. Sample Index")
    axs[1].set_xlabel("Sorted Sample Index")
    axs[1].set_ylabel("Quantile")
    axs[1].legend()
    axs[1].grid(True)
    
    # c) Shifts c vs. index
    axs[2].plot(idx, c_list, 'm.-', label='Shifts c')
    axs[2].set_title("Shift Values c")
    axs[2].set_xlabel("Sorted Sample Index")
    axs[2].set_ylabel("c")
    axs[2].legend()
    axs[2].grid(True)
    
    # d) Link function (h) and convex envelope (h_tilde)
    axs[3].plot(idx, h_list, 'c.-', label='h (link function)')
    axs[3].plot(idx, h_tilde_values, 'kx-', label='h_tilde (convex envelope)')
    axs[3].set_title("Link Function vs. Convex Envelope")
    axs[3].set_xlabel("Sorted Sample Index")
    axs[3].set_ylabel("h-values")
    axs[3].legend()
    axs[3].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # -----------------------
    # 7) Check tester result
    # -----------------------
    # Convert our single bidder samples into shape [ [samples], ... ] 
    # so it matches the function signature
    result = regularized_tester([samples], alpha, delta)
    print("regularized_tester result:", result)

def debug_tester_irregularity():
    """
    Test the regularized_tester function with an irregular distribution
    and visualize the results. Specifically, we will use a mixed distribution and we want to visualize the
    link function and its convex envelope.
    """





# Run the debug visualization
if __name__ == "__main__":
    debug_regularized_tester_once()
