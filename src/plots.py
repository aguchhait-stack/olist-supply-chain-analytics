import matplotlib.pyplot as plt

def plot_top_revenue_categories(data):

    plt.figure(figsize=(10,6))
    plt.barh(y='product_category',width='category_wise_total_order_value',data=data,color='darkblue')
    plt.title("Top 10 Product Categories by Revenue (£)", fontsize=13, fontweight='bold',pad=15)
    plt.grid(alpha=0.3,axis='x')
    plt.xlabel("Revenue (£)")
    plt.tight_layout()
    plt.savefig('outputs.png')
    plt.show()