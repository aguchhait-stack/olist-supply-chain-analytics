import pandas as pd
import matplotlib.pyplot as plt

def plot_monthly_performance_dashboard(logistics_df: pd.DataFrame):
    """
    Plot the monthly business performance dashboard.
    """

    visualization_df = logistics_df.set_index('order_purchase_timestamp').copy()
    # Monthly revenue
    fig, ax = plt.subplots(3,2,figsize=(18,10),sharex=False,sharey=False)
    visualization_df["total_price"].resample('ME').sum().plot(kind='area',alpha=0.3,color='blue',ax=ax[0,0])
    ax[0,0].set_title('Monthly Revenue')
    ax[0,0].set_ylabel('Revenue (R$)')
    ax[0,0].grid(True, alpha=0.3)
    ax[0,0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))

    # Month order
    visualization_df["order_id"].resample('ME').nunique().plot(kind='area',alpha=0.3,color='green',ax=ax[0,1])
    ax[0,1].set_title('Monthly Orders')
    ax[0,1].set_ylabel('Orders')
    ax[0,1].grid(True, alpha=0.3)

    # Top 5 category 
    category_pivot = visualization_df.groupby([visualization_df.index.to_period('M'),'product_category'])['total_price'].sum().unstack().fillna(0)
    top5 = category_pivot.sum().nlargest(5)
    category_pivot[top5.index].plot(kind='area',alpha=0.8, stacked = True, colormap='viridis', ax=ax[1,0])
    ax[1,0].set_title('Top 5 Category by Revenue Share')
    ax[1,0].set_ylabel('Revenue (R$)')
    ax[1,0].grid(True, alpha=0.3)

    # Customer review
    (visualization_df["mean_review_score"].resample('ME').mean().interpolate()).plot(kind='line',alpha=0.8,color='teal',marker= 's', label='Mean Score',ax=ax[1,1])
    ax[1,1].set_title('Average Customer Review Score')
    ax[1,1].set_ylabel('Score')
    ax[1,1].grid(True, alpha=0.3)

    # Delivery days
    (visualization_df["delivery_days"].resample('ME').mean().interpolate()).plot(kind='line',alpha=0.8,marker='d', color='orange',ax=ax[2,0])
    ax[2,0].set_title('Average Delivery Days')
    ax[2,0].set_ylabel('Days')
    ax[2,0].grid(True, alpha=0.3)

    # Late  rate
    (visualization_df["is_late"].resample('ME').mean().interpolate().mul(100)).plot(kind='line',marker='^', color='red',alpha=0.8,ax=ax[2,1])
    ax[2,1].set_title('Average Late rate')
    ax[2,1].set_ylabel('Late Rate (%)')
    ax[2,1].grid(True, alpha=0.3)
    fig.suptitle('Monthly Business Performance Dashboard',fontsize=13, fontweight='bold')
    plt.tight_layout()
    for a in ax.flatten():
        a.set_xlabel('')
    plt.savefig('../outputs/monthly_performance_dashboard.png')
    plt.show()