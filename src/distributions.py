import numpy as np
from scipy.stats import uniform, norm, lognorm
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

np.random.seed(22)

class Distribution(ABC):
    @abstractmethod
    def sample(self, n_samples):
        pass
    
    @abstractmethod
    def cdf(self, x):
        pass
    
    @abstractmethod
    def pdf(self, x):
        pass
    
    def quantile(self, x):
        """Return the reverse quantile function q(x) = 1 - F(x)"""
        return 1 - self.cdf(x)


class UniformDistribution(Distribution):
    def __init__(self, low=0, high=1):
        self.low = low
        self.high = high
        self.dist = uniform(loc=low, scale=high-low)
    
    def sample(self, n_samples):
        return self.dist.rvs(size=n_samples)
    
    def cdf(self, x):
        return self.dist.cdf(x)
    
    def pdf(self, x):
        return self.dist.pdf(x)
    
    def inverse_cdf(self, q):
        """Return x such that F(x) = q"""
        return self.dist.ppf(q)


class EmpiricalDistribution(Distribution):
    def __init__(self, samples):
        self.samples = np.sort(samples)
        self.n = len(samples)
    
    def sample(self, n_samples):
        return np.random.choice(self.samples, size=n_samples, replace=True)
    
    def cdf(self, x):
        """Empirical CDF"""
        return np.sum(self.samples <= x) / self.n
    
    def pdf(self, x):
        """Empirical PDF using kernel density estimation"""
        # Simple implementation for demonstration
        bandwidth = 0.1  # Can be optimized
        kernel_vals = np.exp(-0.5 * ((x - self.samples) / bandwidth) ** 2)
        return np.mean(kernel_vals) / (bandwidth * np.sqrt(2 * np.pi))


class ProductDistribution:
    def __init__(self, distributions):
        self.distributions = distributions
        self.n = len(distributions)
    
    def sample(self, n_samples):
        """Return n_samples from each distribution"""
        return [dist.sample(n_samples) for dist in self.distributions]
    
    def sample_one(self):
        """Return one sample from each distribution"""
        return [dist.sample(1)[0] for dist in self.distributions]
    

class MixtureDistribution(Distribution):
    def __init__(self, dist1, dist2, p1=0.9):
        self.dist1 = dist1  # First distribution
        self.dist2 = dist2  # Second distribution
        self.p1 = p1        # Probability of sampling from dist1
    
    def sample(self, n_samples):
        # Determine how many samples to take from each distribution
        n1 = np.random.binomial(n_samples, self.p1)
        n2 = n_samples - n1
        
        # Sample from each distribution
        samples1 = self.dist1.sample(n1) if n1 > 0 else np.array([])
        samples2 = self.dist2.sample(n2) if n2 > 0 else np.array([])
        
        # Combine and shuffle samples
        samples = np.concatenate([samples1, samples2])
        np.random.shuffle(samples)
        return samples
    
    def cdf(self, x):
        return self.p1 * self.dist1.cdf(x) + (1 - self.p1) * self.dist2.cdf(x)
    
    def pdf(self, x):
        return self.p1 * self.dist1.pdf(x) + (1 - self.p1) * self.dist2.pdf(x)
  


class GaussianDistribution(Distribution):
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std
        self.dist = norm(loc=mean, scale=std)
    
    def sample(self, n_samples):
        return self.dist.rvs(size=n_samples)
    
    def cdf(self, x):
        return self.dist.cdf(x)
    
    def pdf(self, x):
        return self.dist.pdf(x)
    
    def inverse_cdf(self, q):
        """Return x such that F(x) = q"""
        return self.dist.ppf(q)



class LogNormalDistribution(Distribution):
    """
    Log-normal with underlying Normal(μ, σ²).  The SciPy parametrisation is:
        lognorm(s=σ, scale=exp(μ), loc=0)
    so we just store μ and σ and build the frozen rv.
    """
    def __init__(self, mu=0.0, sigma=1.0):
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.mu = mu
        self.sigma = sigma
        self.dist = lognorm(s=sigma, scale=np.exp(mu))  # loc=0 by default

    def sample(self, n_samples):
        return self.dist.rvs(size=n_samples)

    def cdf(self, x):
        return self.dist.cdf(x)

    def pdf(self, x):
        return self.dist.pdf(x)

    def inverse_cdf(self, q):
        """Return x such that F(x)=q (a.k.a. the quantile function)."""
        return self.dist.ppf(q)
    

from tests.base_tester import BaseTester

class UniformTester(BaseTester):
    """Tester for Uniform distributions."""
    
    def __init__(self, seed=22):
        super().__init__(seed)
        self.dist_type = "uniform"
    
    def get_distribution(self, epsilon):
        """Get a uniform mixture distribution with the given epsilon."""
        dist1 = UniformDistribution(0, 1)
        safe_epsilon = max(epsilon, 0.001)  # Avoid division by zero
        dist2_low = (1 / safe_epsilon) ** 2
        dist2_high = dist2_low + 1
        dist2 = UniformDistribution(dist2_low, dist2_high)
        return MixtureDistribution(dist1, dist2, p1=(1-epsilon))
    
    def test_multiple_cases(self, epsilon_values, alpha_values, num_samples=1000):
        """Test multiple cases with uniform distributions."""
        return super().test_multiple_cases(
            self.get_distribution, epsilon_values, alpha_values, num_samples
        )

class GaussianTester(BaseTester):
    """Tester for Gaussian distributions."""
    
    def __init__(self, seed=22):
        super().__init__(seed)
        self.dist_type = "gaussian"
    
    def get_distribution(self, epsilon):
        """Get a Gaussian mixture distribution with the given epsilon."""
        dist1 = GaussianDistribution(1, 1)
        safe_epsilon = max(epsilon, 0.001)
        dist2_mean = 2 ** (1 / safe_epsilon)
        dist2 = GaussianDistribution(dist2_mean, 1)
        return MixtureDistribution(dist1, dist2, p1=(1-epsilon))
    
    def test_multiple_cases(self, epsilon_values, alpha_values, num_samples=1000):
        """Test multiple cases with Gaussian distributions."""
        return super().test_multiple_cases(
            self.get_distribution, epsilon_values, alpha_values, num_samples
        )

class LogNormalTester(BaseTester):
    """Tester for LogNormal distributions."""
    
    def __init__(self, seed=22):
        super().__init__(seed)
        self.dist_type = "lognormal"
    
    def get_distribution(self, epsilon):
        """Get a LogNormal mixture distribution with the given epsilon."""
        dist1 = LogNormalDistribution(0, 1)
        dist2_mu = np.log(2 ** (1 / max(epsilon, 0.001)))
        dist2 = LogNormalDistribution(dist2_mu, 1)
        return MixtureDistribution(dist1, dist2, p1=(1-epsilon))
    
    def test_multiple_cases(self, epsilon_values, alpha_values, num_samples=1000):
        """Test multiple cases with LogNormal distributions."""
        return super().test_multiple_cases(
            self.get_distribution, epsilon_values, alpha_values, num_samples
        )