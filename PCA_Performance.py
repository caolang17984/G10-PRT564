# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.stats import kstest
import math
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import mode
from sklearn.metrics import silhouette_score
# Part 0: INPUT DATA

# read a remote .csv file
df = pd.read_csv('df_model.csv')
df = df[['Class','Cited by','Title Length','Country_Count','Author_Count','Institution_Count','Journal_encodeder','Publisher_encodeder','Rank_group']] 
exploratory_vars = df.iloc[:,2:-1]
print(exploratory_vars)



# Part I: explore PCA components of the dataset
# Find the good number of component for each model based on 
# the threshold of PCA according to 90%, 95%, & 99%
def PCA_components(data, n_components=None):
    # Perform PCA
    pca = PCA(n_components=n_components)
    
    projected_data = pca.fit_transform(data)
    
    # compare the cumulative explained variance versus number of PCA components
    pca = PCA().fit(data)
    
    plt.figure(figsize=(8, 6), facecolor='#F2EEE5')  # Set background color of the figure
    plt.scatter(projected_data[:, 0], projected_data[:, 1],
                c=range(len(projected_data)), edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap('Spectral', len(projected_data)))
    plt.title('Principle Component Analysis of PPCs', fontweight='bold')
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()
    plt.gca().set_facecolor('#F2EEE5')  # Set background color inside the plot
    plt.show()
    
    
    # Plot cumulative explained variance versus number of components
    plt.figure(figsize=(8, 6), facecolor='#F2EEE5')  # Set background color of the figure
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.title('Cumulative explained variance of the PPC versus the number of PCA components', fontweight='bold')
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.gca().set_facecolor('#F2EEE5')  # Set background color inside the plot
    plt.show()
    
    print("Without Standardisation")
    # Determine number of components required to explain the specified percentage of variance
    explained_variance_thresholds=[0.90, 0.95, 0.99]
    for threshold in explained_variance_thresholds:
        # Determine number of components required to explain the specified percentage of variance
        pca = PCA(threshold).fit(data)
        print("%.0f%% variance is explained by: %d components." % ((threshold * 100), pca.n_components_))
        
def PCA_components_standardisation(data, n_components=None):
    # Data preprocessing: Standardization
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(data)
    
    # Perform PCA
    pca = PCA(n_components=n_components)    
    projected_data = pca.fit_transform(df_scaled)
    
    # compare the cumulative explained variance versus number of PCA components
    pca = PCA().fit(df_scaled)
          
    plt.figure(figsize=(8, 6), facecolor='#F2EEE5')  # Set background color of the figure
    plt.scatter(projected_data[:, 0], projected_data[:, 1],
                c=range(len(projected_data)), edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap('Spectral', len(projected_data)))
    plt.title('Principle Component Analysis of PPCs', fontweight='bold')
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()
    plt.gca().set_facecolor('#F2EEE5')  # Set background color inside the plot
    plt.show()
    
    
    # Plot cumulative explained variance versus number of components
    plt.figure(figsize=(8, 6), facecolor='#F2EEE5')  # Set background color of the figure
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.title('Cumulative explained variance of the PPC versus the number of PCA components', fontweight='bold')
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.gca().set_facecolor('#F2EEE5')  # Set background color inside the plot
    plt.show()
    
    
    print("With Standardisation")
    # Determine number of components required to explain the specified percentage of variance
    explained_variance_thresholds=[0.90, 0.95, 0.99]
    
    for threshold in explained_variance_thresholds:
        # Determine number of components required to explain the specified percentage of variance
        pca = PCA(threshold).fit(df_scaled)
        print("%.0f%% variance is explained by: %d components." % ((threshold * 100), pca.n_components_))

# Part II: Create Dataset with PCA components, explored above
# Ojective: 
# Obj_1: Finallize the list exploratory variables have strong influence to extracted papers.
# Obj_2: Define the common function to 
def perform_pca(df, file_name, n_components):
    # Data preprocessing: Standardization
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    
   # Perform PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df_scaled)
    # print("Show Principle Components")
    # print(abs(pca.components_))
    
    result_df = pd.DataFrame(data=abs(pca.components_))
    result_df.columns = df.columns
    # file_name = str(n_components)+file_name
    result_df.to_csv(file_name, index=False)
    
    # Create a new DataFrame from the principal components
    principal_df = pd.DataFrame(data=principal_components, 
                                columns=[f'PC{i}' for i in range(1, n_components+1)])
    
    # Print explained variance ratio by each principal component
    pca_ns = PCA().fit(df)
    pca_s = PCA().fit(df_scaled)
    explained_variance_ratio = pca.explained_variance_ratio_
    print('Explained variation per principal component (no standardisation): {}'.format(pca_ns.explained_variance_ratio_))
    print('Explained variation per principal component (standardisation): {}'.format(pca_s.explained_variance_ratio_))
    for i, variance in enumerate(explained_variance_ratio):
        print(f"Principal Component {i+1}: {variance}")
    return principal_df


# Find the number of component that 90%, 95%, and 99% variance
exploratory_vars.to_csv('test.csv')
PCA_components(exploratory_vars,n_components=5)
PCA_components_standardisation(exploratory_vars,n_components=5)


print('PCA performance')
PCA_perform_2 = perform_pca(exploratory_vars, pca_performance_filepath, n_components=7)
