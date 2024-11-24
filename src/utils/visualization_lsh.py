import numpy as np
import matplotlib.pyplot as plt

def plot_s_curves(fixed_r, fixed_b, s_range):
    """
    Plots S-curves for two configurations: varying b with fixed r and varying r with fixed b.
    
    Args:
        fixed_r: The fixed r value to plot S-curves while varying b.
        fixed_b: The fixed b value to plot S-curves while varying r.
        s_range: Range of similarity values (0 to 1).
    """
    s_values = np.linspace(s_range[0], s_range[1], 100)

    # Creating a 1x2 plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plotting for fixed r, varying b (Left Plot)
    for b in range(1, 51, 5):
        prob = 1 - (1 - s_values**fixed_r)**b
        axes[0].plot(s_values, prob, label=f'b={b}')
    axes[0].axvline(x=0.7, color='black', linestyle='--', label='Threshold 0.7')
    axes[0].set_title(f'S-curve for fixed r={fixed_r}, varying b')
    axes[0].set_xlabel('Similarity')
    axes[0].set_ylabel('Prob(Candidate Pair)')
    axes[0].legend()

    # Plotting for fixed b, varying r (Right Plot)
    for r in range(1, 11, 1):
        prob = 1 - (1 - s_values**r)**fixed_b
        axes[1].plot(s_values, prob, label=f'r={r}')
    axes[1].axvline(x=0.7, color='black', linestyle='--', label='Threshold 0.7')
    axes[1].set_title(f'S-curve for fixed b={fixed_b}, varying r')
    axes[1].set_xlabel('Similarity')
    axes[1].set_ylabel('Prob(Candidate Pair)')
    axes[1].legend()

    plt.tight_layout()
    plt.show()