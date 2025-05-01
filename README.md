# Justin Dominic's Senior Thesis
# Beyond Regularity: Stronger Revenue Guarantees with MHR Distributions in Testable Learning

This repository contains an implementation of the regularized testable learning framework for mechanism design, specifically for learning revenue-optimal auctions from samples that may be adversarially corrupted.

## Project Overview

This project implements a framework for robust learning of optimal auctions, inspired by the paper "Safely Learning Optimal Auctions" and other related works. The key insight is creating a testing framework that can determine when theorems and learning algorithms for revenue-maximizing auctions can be safely applied to data, even when the data may not perfectly match the underlying theoretical assumptions.

### Key Components

- **Regularized Tester**: Determines if a given distribution is close enough to a regular distribution to apply theorems that require regularity
- **Regularized Learner**: Learns near-optimal reserve prices for auctions, even when the input data may be corrupted
- **Distribution Models**: Implementations of various distributions (uniform, normal, log-normal, etc.) for testing and evaluation
- **Visualization Tools**: Functions to visualize the algorithm's performance, decisions, and relevant distribution properties

### Theoretical Background

The project focuses on the problem of learning revenue-optimal auctions when:
1. The bidders' valuation data may be corrupted or drawn from distributions that are adversarially perturbed
2. The distributions may not satisfy the regularity condition that is commonly assumed in auction theory

The algorithms provide guarantees relative to the optimal revenue of any regular distribution that is close to the given data in Kolmogorov-Smirnov distance.

## Repository Structure

- `src/`: Source code for the main algorithms and utilities
  - `algorithms.py`: Implementation of the regularized tester and learner
  - `distributions.py`: Classes for various distribution types
  - `illustrations.py`: Utility functions for visualizing regularization concepts
  - `utils.py`: Helper functions for computations and visualizations

- `tests/`: Test scripts to evaluate the algorithms
  - `base_tester.py`: Base class for testing the algorithms with different distributions
  - `test_single_bidder.py`: Tests for the single-bidder setting
  - `test_suite.py`: Running tests with different distribution types
  - `test_tester.py`: Tests specifically focusing on the tester's decisions

## Installation and Setup

1. Clone the repository:
```bash
git clone https://github.com/jcdominic/safely-learning-optimal-auctions.git
cd safely-learning-optimal-auctions
```
**Or** download as a zip file.

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Make sure you have the following directory structure for saving plots:
```bash
mkdir -p plots/debugging/{uniform,gaussian,lognormal}
```

## Running the Code

### Basic Tests

To run the basic tests with a single bidder and uniform distribution:

```bash
python -m tests/test_single_bidder.py
```

### Testing Across Different Distributions

To run tests with specific distribution types:

```bash
python -m tests/test_suite.py
```

You can modify the code to use different distribution types ('uniform', 'gaussian', or 'lognormal') and adjust the parameters.

### Testing Tester Robustness

To analyze how the tester behaves with different epsilon/alpha ratios:

```bash
python -m tests/test_tester.py
```

This will generate visualizations showing when the tester says YES vs NO for different corruption and tolerance parameters.

### Customizing Tests

You can customize the tests by modifying the parameters in the test scripts:

- `epsilon_values`: Controls the level of corruption in the distribution (higher = more irregular)
- `alpha_values`: The distortion parameter used by the algorithms (higher = more tolerance)
- `num_samples`: Number of samples to use for each test

For example, to run a test with custom parameters:

```python
from tests.test_suite import run_tests

custom_epsilons = [0.01, 0.1, 0.5]
custom_alphas = [0.01, 0.1, 0.5]
results = run_tests('lognormal', custom_epsilons, custom_alphas, num_samples=5000)
```

## Understanding the Algorithms

### Regularized Tester (Algorithm 1)

The tester decides whether a distribution is close enough to a regular distribution:

1. Takes samples from a distribution and a distortion parameter α
2. Constructs a quantile-shifted version of the empirical distribution
3. Applies the convex envelope to the link function to ensure regularity
4. Checks if the resulting distribution is within α-distance of the original
5. Returns "YES" if there exists a close regular distribution, "NO" otherwise

### Regularized Learner (Algorithm 2)

The learner finds a near-optimal reserve price:

1. Takes samples from a distribution and a distortion parameter α
2. Constructs a quantile-shifted version of the empirical distribution
3. Applies the convex envelope to the link function to ensure regularity
4. Further shifts the quantiles to handle potential corruptions
5. Returns the reserve price that maximizes revenue for this conservative estimate

## Key Theoretical Results

The implemented algorithms achieve:

1. For Monotone Hazard Rate (MHR) distributions with corruption level α:
   - Revenue approximation ratio of 1-O(α)

2. For Regular distributions with corruption level α:
   - Revenue approximation ratio of 1-O(√α)

These bounds are tight, as shown by the lower bound tests in the codebase.

## Extending the Project

### Adding New Distribution Types

To add a new distribution type:

1. Add a new distribution class in `src/distributions.py`
2. Implement the required methods: `sample()`, `cdf()`, `pdf()`, etc.
3. Add a corresponding tester class inheriting from `BaseTester`

### Implementing Multi-Bidder Auctions

The current implementation focuses mainly on single-bidder auctions. To extend to multi-bidder:

1. Modify the `regularized_learner` to compute optimal reserve prices for multiple bidders
2. Implement Myerson's auction mechanism for the multi-bidder case
3. Create a revenue evaluator that handles multiple bidders

## Debugging and Visualization

The project includes various visualization tools:

- `plot_distribution_diagnostics()`: Visualizes PDF, CDF, and quantile functions
- `verify_irregularity()`: Plots virtual value functions to check regularity
- `plot_tester_decisions_vs_epsilon_alpha_ratio()`: Analyzes tester decisions

Plots are saved in the `plots/` directory, with debugging visualizations in subdirectories.

## References

This project is based on concepts from the following papers:

- "Safely Learning Optimal Auctions: A Testable Learning Framework for Mechanism Design"
- "Robust Learning of Optimal Auctions" (Guo, Jordan, Zampetakis)
- "Settling the Sample Complexity of Single-parameter Revenue Maximization" (Guo, Huang, Zhang)