import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score        

def preprocess_data(df):    
    """
    Preprocess the TCGA data.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing TCGA data.
    
    Returns:
    pd.DataFrame: Preprocessed DataFrame.
    """
    # Handle missing values
    df = df.dropna()
    
    # Convert categorical variables to numerical
    df = pd.get_dummies(df, drop_first=True)
    
    return df   

def load_data(file_path):
    """
    Load TCGA data from a CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file containing TCGA data.
    
    Returns:
    pd.DataFrame: DataFrame containing the loaded data.
    """
    return pd.read_csv(file_path)

def cluster_data(df, n_clusters=3):
    """
    Cluster the TCGA data using KMeans.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing TCGA data.
    n_clusters (int): Number of clusters to form.
    
    Returns:
    pd.Series: Cluster labels for each sample.
    """
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(scaled_data)
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(reduced_data)
    
    return pd.Series(cluster_labels, index=df.index)

def evaluate_clustering(df, cluster_labels):
    """
    Evaluate the clustering using silhouette score.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing TCGA data.
    cluster_labels (pd.Series): Cluster labels for each sample.
    
    Returns:
    float: Silhouette score of the clustering.
    """
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    # Calculate silhouette score
    score = silhouette_score(scaled_data, cluster_labels)
    
    return score

def main(file_path, n_clusters=3):
    """
    Main function to load data, preprocess, cluster, and evaluate the clustering.
    
    Parameters:
    file_path (str): Path to the CSV file containing TCGA data.
    n_clusters (int): Number of clusters to form.
    """
    # Load data
    df = load_data(file_path)
    
    # Preprocess data
    df = preprocess_data(df)
    
    # Cluster data
    cluster_labels = cluster_data(df, n_clusters)
    
    # Evaluate clustering
    score = evaluate_clustering(df, cluster_labels)
    
    print(f'Silhouette Score: {score}')
    print(f'Cluster Labels:\n{cluster_labels.value_counts()}')  

if __name__ == "__main__":
    # Example usage
    import os
    tcga_data_dir = os.getcwd() # Replace with your actual directory path
    file_path = 'transformed_tcga.csv'  # Replace with your actual file path
    file_path = os.path.join(tcga_data_dir, file_path)
    main(file_path, n_clusters=2)
