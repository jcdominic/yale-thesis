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

        empirical_cdf = np.arange(m) / m
        
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

        
        # Apply link function to all points 
        # Pg 4 -- Conv(D) denotes the convex envelope of the distribution's link function
        h_values = []
        for q_hat in q_hat_values:
            try:
                h_values.append(link_function(1 - q_hat))
            except:
                # Handle division by zero or other numerical issues
                h_values.append(float('inf'))
        
        # Compute convex envelope of all points together
        h_tilde_values = convex_envelope(bidder_samples[bidder_samples > 0], h_values)
        
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
            if diff > (c + 0.00000001): # adding very small bufer for floating point errors
                return "NO"
    
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