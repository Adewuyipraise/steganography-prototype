"""Result plotting and visualization"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_results(original, stego, ber_results):
    """
    Visualize original and stego images, and BER under attacks.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(original)
    axes[0].set_title("Original Cover")
    axes[1].imshow(stego)
    axes[1].set_title("Stego Image")
    plt.tight_layout()
    plt.show()

    if ber_results:
        df = pd.DataFrame(list(ber_results.items()), columns=['Attack', 'BER'])
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x='Attack', y='BER')
        plt.title("Bit Error Rate (BER) under different attacks")
        plt.ylabel("BER")
        plt.xticks(rotation=45)
        plt.show()
