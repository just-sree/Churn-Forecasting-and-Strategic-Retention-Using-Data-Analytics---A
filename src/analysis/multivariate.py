import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import io
from datetime import datetime
import warnings
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')

def multivariate_analysis(df, dataset_name, figsize=(12, 8), save_results=True):
    """
    Perform multivariate analysis on the dataset including correlations, PCA, anomaly detection, clustering, and outlier analysis.
    
    Parameters:
    df : pandas DataFrame
        The dataset to analyze
    dataset_name : str
        Name of the dataset for directory and file naming
    figsize : tuple
        Size of the plots
    save_results : bool
        Whether to save results to files
    """
    
    # Create output directories if saving results
    if save_results:
        current_file_path = os.path.abspath(__file__)
        src_dir = os.path.dirname(os.path.dirname(current_file_path))
        project_root = os.path.dirname(src_dir)
        
        dataset_dir_name = dataset_name.lower().replace(' ', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reports_dir = os.path.join(project_root, 'reports', 'multivariate_analysis')
        base_dir = os.path.join(reports_dir, dataset_dir_name, timestamp)
        plots_dir = os.path.join(base_dir, 'plots')
        stats_dir = os.path.join(base_dir, 'statistics')
        
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(stats_dir, exist_ok=True)
        
        log_file = open(os.path.join(base_dir, "analysis_log.txt"), 'w')
        sys.stdout = log_file
    
    print(f"\n{'='*80}\n{dataset_name.upper()} MULTIVARIATE ANALYSIS\n{'='*80}")
    print(f"Analysis performed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Outlier Analysis
    plt.figure(figsize=figsize)
    sns.boxplot(data=df.select_dtypes(include=['float64', 'int64']))
    plt.title('Outlier Analysis')
    if save_results:
        plt.savefig(os.path.join(plots_dir, "outlier_analysis.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Correlation Matrix
    plt.figure(figsize=figsize)
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    if save_results:
        plt.savefig(os.path.join(plots_dir, "correlation_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. PCA
    numeric_df = df.select_dtypes(include=['int64', 'float64']).dropna()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)
    pca = PCA(n_components=2)
    df_pca = pd.DataFrame(pca.fit_transform(scaled_data), columns=['PC1', 'PC2'])
    
    plt.figure(figsize=figsize)
    sns.scatterplot(x=df_pca['PC1'], y=df_pca['PC2'], alpha=0.6)
    plt.title('PCA - First Two Principal Components')
    if save_results:
        plt.savefig(os.path.join(plots_dir, "pca_scatter.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Anomaly Detection
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    df['Anomaly_Score'] = iso_forest.fit_predict(scaled_data)
    plt.figure(figsize=figsize)
    sns.scatterplot(x=df_pca['PC1'], y=df_pca['PC2'], hue=df['Anomaly_Score'], palette={1: 'blue', -1: 'red'})
    plt.title('Anomaly Detection (Red: Anomalies)')
    if save_results:
        plt.savefig(os.path.join(plots_dir, "anomaly_detection.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled_data)
    plt.figure(figsize=figsize)
    sns.scatterplot(x=df_pca['PC1'], y=df_pca['PC2'], hue=df['Cluster'], palette='viridis')
    plt.title('K-Means Clustering')
    if save_results:
        plt.savefig(os.path.join(plots_dir, "kmeans_clustering.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Reset stdout
    if save_results:
        sys.stdout = sys.__stdout__
        log_file.close()
    
    print(f"Analysis results saved to {base_dir}/")
    print(f"Multivariate analysis completed successfully.")
    
# Example usage
if __name__ == "__main__":
    dataset_name = "Telecom Churn Dataset"
    df = pd.read_csv("hf://datasets/aai510-group1/telco-customer-churn/train.csv")
    multivariate_analysis(df, dataset_name)
