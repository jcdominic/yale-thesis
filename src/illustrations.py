import numpy as np
import matplotlib.pyplot as plt
from src.utils import convex_envelope

def virtual_value(pdf, cdf, v):
    """Compute virtual value φ(v) = v - (1-F(v))/f(v);
       pdf, cdf are callables defined on v (numpy array)."""
    f = pdf(v)
    F = cdf(v)
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = v - (1 - F) / f
        phi[np.isinf(phi)] = np.nan
    return phi

def visualize_regularity_and_irregularity():
    # ---------------------------
    # Regular distributions (LHS)
    # ---------------------------
    # Uniform[0,1]
    x_u = np.linspace(0, 1, 400)
    pdf_u  = lambda x: np.ones_like(x)
    cdf_u  = lambda x: x
    phi_u  = virtual_value(pdf_u, cdf_u, x_u)

    # Exponential(λ=1)
    x_e = np.linspace(0, 5, 800)
    pdf_e = lambda x: np.exp(-x)
    cdf_e = lambda x: 1 - np.exp(-x)
    phi_e = virtual_value(pdf_e, cdf_e, x_e)

    # -----------------------------
    # Irregular distributions (RHS)
    # -----------------------------
    # Mixture of uniforms: ½·U[0,0.4] + ½·U[0.6,1]
    x_m = np.linspace(0, 1, 600)
    def pdf_m(x):
        return (
            1.25 * ((0 <= x) & (x <= 0.4)).astype(float)
            + 1.25 * ((0.6 <= x) & (x <= 1)).astype(float)
        )

    def cdf_m(x):
        x = np.asarray(x)
        res = np.zeros_like(x)
        # first block
        mask1 = (0 <= x) & (x <= 0.4)
        res[mask1] = 1.25 * x[mask1]
        # gap 0.4-0.6
        mask2 = (0.4 < x) & (x < 0.6)
        res[mask2] = 1.25 * 0.4
        # second block
        mask3 = x >= 0.6
        res[mask3] = 1.25 * 0.4 + 1.25 * (x[mask3] - 0.6)
        return res

    phi_m = virtual_value(pdf_m, cdf_m, x_m)

    # clip extreme values for nicer plotting
    phi_m_clip = np.clip(phi_m, -10, 5)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex='col')

    # Left column – regular
    axes[0, 0].plot(x_u, pdf_u(x_u), label="Uniform[0,1]")
    axes[0, 0].plot(x_e, pdf_e(x_e), label="Exponential(λ=1)")
    axes[0, 0].set_title("Regular distributions")
    axes[0, 0].set_ylabel("PDF")
    axes[0, 0].legend()

    axes[1, 0].plot(x_u, phi_u, label="Uniform φ(v)")
    axes[1, 0].plot(x_e, phi_e, label="Exponential φ(v)")
    axes[1, 0].axhline(0, color="gray", lw=0.8)
    axes[1, 0].set_title("Monotone virtual values")
    axes[1, 0].set_xlabel("v")
    axes[1, 0].set_ylabel("Virtual value φ(v)")
    axes[1, 0].legend()

    # Right column – irregular
    # axes[0, 1].plot(x_t, pdf_t(x_t), label="Triangular")
    axes[0, 1].plot(x_m, pdf_m(x_m), label="Mixture uniforms")
    axes[0, 1].set_title("Irregular distributions")
    axes[0, 1].legend()

    # axes[1, 1].plot(x_t, phi_t_clip, label="Triangular φ(v)")
    axes[1, 1].plot(x_m, phi_m_clip, label="Mixture φ(v)")
    axes[1, 1].axhline(0, color="gray", lw=0.8)
    axes[1, 1].set_ylim(-10, 5)
    axes[1, 1].set_title("Non‑monotone virtual values")
    axes[1, 1].set_xlabel("v")
    axes[1, 1].legend()

    plt.tight_layout()
    file_path = "plots/regular_vs_irregular_graphic.png"
    fig.savefig(file_path, dpi=300)


def plot_sin_with_convex_envelope(num_points=400):
    x = np.linspace(0, 2 * np.pi, num_points)
    y = np.sin(x ** 1.4) / (1 + 0.3 * x)
    y_env = convex_envelope(x, y)

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label='Original Function')
    plt.plot(x, y_env, linestyle='--', label='Convex Envelope')
    plt.title('Visualizing the Convex Envelope')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.show()
    # file_path = "plots/regular_vs_irregular_graphic.png"
    # fig.savefig(file_path, dpi=300)


if __name__ == "__main__":
    plot_sin_with_convex_envelope()
