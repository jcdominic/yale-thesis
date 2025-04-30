import numpy as np
import matplotlib.pyplot as plt
from utils import convex_envelope, link_function, inverse_link_function

# Create x-values and a non-convex y-function
x = np.linspace(0, 10, 100)
y = np.sin(x) + 0.2 * x  # not convex

# Compute convex envelope
y_env = convex_envelope(x, y)

# TEST INVERSE LINK -- Convert to quantiles and plot
q_tilde_values = []
for y_val in y_env:
    q_tilde = 1 - inverse_link_function(link_function(y_val))
    q_tilde_values.append(q_tilde)
  

# Plot original function and its convex envelope
plt.figure(figsize=(8, 5))
plt.plot(x, y, label='Original Function', color='blue', alpha=0.6)
plt.plot(x, y_env, label='Convex Envelope', color='red', linestyle='--', linewidth=2)
plt.plot(x, q_tilde_values, label='Quantile Function', color='green', linestyle=':', linewidth=2)
plt.fill_between(x, y, y_env, where=(y > y_env), color='red', alpha=0.2, label='Gap (non-convex region)')
plt.title('Convex Envelope of a Non-Convex Function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
