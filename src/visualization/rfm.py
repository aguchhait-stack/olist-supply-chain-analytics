import pandas as pd
import matplotlib.pyplot as plt

def plot_rfm_segments(rfm: pd.DataFrame):
    """
    Plot RFM customer segments scatter plot.
    """
    plt.figure(figsize=(10, 6))
    colors = {'Champions':'red','Promising':'blue','Lost':'green','At Risk':'orange'}
    for segment, color in colors.items():
        subset = rfm[rfm["KMeans_Segment_name"] == segment]
        plt.scatter(subset['recency'],subset['monetary'],c=color,
                    s=subset['frequency']*25, alpha = 0.3, label=segment)   
    plt.xlabel('Recency (days since last purchase)')
    plt.ylabel('Monetary (total spend R$)')
    plt.title('KMeans Customer Segments by RFM', fontsize=11, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('../outputs/rfm_scatter.png')
    plt.show()