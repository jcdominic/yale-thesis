from src.distributions import UniformTester, GaussianTester, LogNormalTester

def run_tests(distribution_type, epsilon_values=None, alpha_values=None, num_samples=1000):
    """
    Run tests with the specified distribution type.
    
    Args:
        distribution_type: 'uniform', 'gaussian', or 'lognormal'
        epsilon_values: List of epsilon values to test (default: [0.01, 0.05, 0.1, 0.2, 0.3])
        alpha_values: List of alpha values to test (default: [0.01, 0.05, 0.1, 0.2, 0.5])
        num_samples: Number of samples for each test
        
    Returns:
        Results of the tests
    """
    if epsilon_values is None:
        epsilon_values = [0.01, 0.05, 0.1, 0.2, 0.3]
    if alpha_values is None:
        alpha_values = [0.01, 0.05, 0.1, 0.2, 0.5]
    
    if distribution_type.lower() == 'uniform':
        tester = UniformTester()
    elif distribution_type.lower() == 'gaussian':
        tester = GaussianTester()
    elif distribution_type.lower() == 'lognormal':
        tester = LogNormalTester()
    else:
        raise ValueError("Invalid distribution type. Choose 'uniform', 'gaussian', or 'lognormal'.")
    
    return tester.test_multiple_cases(epsilon_values, alpha_values, num_samples)

def test_single_case(distribution_type, epsilon=0.1, alpha=0.01, num_samples=1000):
    """
    Test a single case with the specified distribution type.
    
    Args:
        distribution_type: 'uniform', 'gaussian', or 'lognormal'
        epsilon: Corruption parameter
        alpha: Distortion parameter
        num_samples: Number of samples
        
    Returns:
        Results of the test
    """
    if distribution_type.lower() == 'uniform':
        tester = UniformTester()
    elif distribution_type.lower() == 'gaussian':
        tester = GaussianTester()
    elif distribution_type.lower() == 'lognormal':
        tester = LogNormalTester()
    else:
        raise ValueError("Invalid distribution type. Choose 'uniform', 'gaussian', or 'lognormal'.")
    
    distribution = tester.get_distribution(epsilon)
    return tester.test_single_case(distribution, epsilon, alpha, num_samples)


if __name__ == "__main__":
    # result = test_single_case('lognormal', epsilon=0.1, alpha=0.01, num_samples=100000)

    # Run tests with multiple epsilon and alpha values
    results = run_tests('gaussian', num_samples=100000)

    # Run tests with custom epsilon and alpha values
    # custom_epsilons = [0.01, 0.1, 0.5]
    # custom_alphas = [001, 0.1, 0.5]
    # results = run_tests('lognormal', custom_epsilons, custom_alphas, num_samples=5000)