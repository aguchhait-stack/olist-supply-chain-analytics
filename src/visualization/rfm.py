import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def plot_rfm_segments(rfm: pd.DataFrame):
    """
    Plot RFM customer segments scatter plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'Champions':'red','Promising':'blue','Lost':'green','At Risk':'orange'}
    for segment, color in colors.items():
        subset = rfm[rfm["KMeans_Segment_name"] == segment]
        ax.scatter(subset['recency'],subset['monetary'],c=color,
                    s=subset['frequency']*25, alpha = 0.3, label=segment)   
    ax.set_xlabel('Recency (days since last purchase)')
    ax.set_ylabel('Monetary (total spend R$)')
    ax.set_title('KMeans Customer Segments by RFM', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig('outputs/rfm_scatter.png')
    return fig

def plot_rfm_cluster_validation(X_train_scaled, X_test_scaled, n_clusters_range=range(2, 12)):
    """
    Plot Elbow and Silhouette scores for RFM cluster validation.
    """
    # Evaluate candidate values of k using Elbow and Silhouette methods
    inertia = []
    silhouette =[]
    for cluster in n_clusters_range: # Silhouette need at least two cluster.
        kmeans = KMeans(n_clusters = cluster,random_state = 42)
        kmeans.fit(X_train_scaled) 

        # Within-cluster sum of squares (Elbow)
        inertia.append(kmeans.inertia_)

        # Evaluate clustering on the held-out test set
        test_label = kmeans.predict(X_test_scaled) # Predicted lables
        silhouette.append(silhouette_score(X_test_scaled,test_label)) 

    # Full number fo clusters    
    clusters = list(n_clusters_range)    

    # Plot cluster validation metrics
    fig,axis = plt.subplots(1,2,figsize=(10,5),sharex=True)

    # Elbow Curve
    axis[0].plot(clusters,inertia,marker='D',color='teal')
    axis[0].set_xlabel("Number of Clusters (k)")
    axis[0].set_ylabel("Inertia")
    axis[0].set_title("Elbow Curve")
    axis[0].set_xticks(clusters)
    axis[0].axvline(x=4,color='r',linestyle='--',alpha=0.5,label="Tight Cluster, k = 4")
    axis[0].legend()
    axis[0].grid(alpha=0.3)

    # Silhouette Score
    axis[1].plot(clusters,silhouette,marker='D',color='blue')
    axis[1].set_xticks(clusters)
    axis[1].axvline(x=4,color='r',linestyle='--',alpha=0.5,label="Well Seperated Cluster, k = 4")
    axis[1].legend()
    axis[1].grid(alpha=0.3)
    axis[1].set_title("Silhouette Score")
    axis[1].set_xlabel("Number of Clusters (k)")
    axis[1].set_ylabel("Silhouette")
    plt.suptitle("RFM Cluster Validation", fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/elbow_silhouette_validation.png')
    return fig