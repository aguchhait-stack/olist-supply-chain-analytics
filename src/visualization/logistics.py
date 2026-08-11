import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_sla_breach_by_state(logistics_df, contingency, chi2, p, dof):
    """
    Plot SLA breach rates by customer state against the national rate.
    """
    plt.figure(figsize=(12, 6))
    state_sla_rates = (contingency[1]*100/(contingency[0]+contingency[1])).sort_values(ascending=True)
    national_sla_rate = logistics_df["is_late"].mean()*100
    colors = ['coral' if rate>national_sla_rate else 'teal' for rate in state_sla_rates]
    (state_sla_rates).plot(kind='barh',width=0.85,color=colors,edgecolor='white')
    plt.axvline(national_sla_rate, color='r',linestyle='--',alpha=0.7,
                label = f"National Rate: {national_sla_rate:.2f}%\nChi2:{chi2:.0f}, p={p:.2e}, dof={dof}" )
    plt.xticks(np.arange(0,25,2))
    plt.title('SLA Breach Rate by Customer State (%)', fontsize=11, fontweight='bold')
    plt.xlabel('Late Orders (%)')
    plt.ylabel('Customer State')
    plt.grid(alpha=0.3,axis='x')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/sla_breach_by_state.png')
    plt.show()

def plot_logistics_correlation(logistics_df: pd.DataFrame):
    """
    Plot the Spearman correlation matrix for logistics features.
    """
    logistics_feature = ['is_interstate','is_late','vendor_handling_days','carrier_transit_days','distance_km','mean_review_score','total_freight_value']
    corrrelation_matrix = logistics_df[logistics_feature].corr(method='spearman')
    plt.figure(figsize=(12, 8))
    sns.heatmap(corrrelation_matrix,cmap='coolwarm',square=True,annot=True, vmin=-1, vmax=1, linewidths=0.5, fmt = ".2f")
    plt.title('Logistics Drivers: Spearman Correlation Matrix', fontsize=11, fontweight='bold')
    plt.xticks(rotation=20)
    plt.savefig('outputs/logistics_correlation_matrix.png')
    plt.show()

def plot_logistics_bottlenecks(logistics_df: pd.DataFrame):
    """
    Plot key logistics bottleneck indicators.
    """
    fig ,axis = plt.subplots(1,2,figsize=(14,5))
    # transit days vs vender days
    logistics_df[['carrier_transit_days','vendor_handling_days']].mean().plot(kind='bar',ax= axis[0], color=['teal', 'salmon'])
    axis[0].tick_params(axis= 'x', rotation = 0)
    axis[0].set_title('Avg Vendor Handling vs Carrier Transit')
    axis[0].set_ylabel('Days')
    axis[0].set_yticks(np.arange(0,10,1))
    axis[0].grid(True, alpha=0.3)

    # Interstate vs Intrastate
    colors = {0:'teal',1:'red'}
    for v, c in colors.items():
        subset = logistics_df[logistics_df['is_interstate'] == v]
        subset.plot(x='distance_km',y='total_freight_value', color = c, kind='scatter',alpha=0.3, ax = axis[1], label=f'Interstate: {v}')
    axis[1].set_title('Distance vs Freight Cost')
    axis[1].set_xlabel('Distance (km)')
    axis[1].set_ylabel('Freight Cost (R$)')
    axis[1].set_xlim([0, 4000])
    axis[1].legend(['Intrastate', 'Interstate'])
    axis[1].grid(True, alpha=0.3)
    plt.suptitle('Logistics Bottleneck Analysis', fontsize=11, fontweight='bold')
    plt.savefig('outputs/distance_vs_freight.png')
    plt.tight_layout()
    plt.show()