import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram

'''
Generate relevant correlation data and graphs 
'''

cleaned_dataset = pd.read_csv('cleaned_dataset_final.csv')


# Calculate correlation matrix
correlation_matrix = cleaned_dataset.corr()

# Sort rows and columns based on hierarchical clustering
clustered_corr_matrix = sns.clustermap(correlation_matrix, cmap='coolwarm', annot=False, figsize=(10, 10), linewidths=0.5, linecolor='white', square=False)

# Get the linkage matrix
linkage_matrix = clustered_corr_matrix.dendrogram_col.linkage

# Get the reordered indices
reordered_indices = clustered_corr_matrix.dendrogram_col.reordered_ind

# Map the reordered indices to the column names
labels = [correlation_matrix.columns[i] for i in reordered_indices]

# Plot the dendrogram
plt.figure(figsize=(12, 15))
dendrogram(linkage_matrix, labels=labels, leaf_rotation=90)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Features in cleaned_dataset')
plt.ylabel('Euclidian Distance')
plt.subplots_adjust(bottom=0.3)
plt.savefig('full_dendrogram_final.png')

# Get the reordered index based on clustering
reordered_index = clustered_corr_matrix.dendrogram_row.reordered_ind

# Reorder the correlation matrix based on the clustered index
sorted_corr_matrix = correlation_matrix.iloc[reordered_index, reordered_index]

# Plot the sorted correlation matrix
plt.figure(figsize=(10, 10))
sns.heatmap(sorted_corr_matrix, cmap='coolwarm', annot=False, linewidths=0.5, linecolor='white', square=True, cbar_kws={"orientation": "vertical", "pad": 0.02})
plt.title("Sorted Correlation Matrix", size = 20)
plt.subplots_adjust(bottom=0.1, top=0.9, left=0.15, right=0.85)
# Save the figure
plt.savefig("corr_matrix_all.png", bbox_inches='tight')

# Select upper triangular matrix (excluding diagonal and redundant values)
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

# Find the top 5 highest correlated features
top_correlated_features = upper_triangle.unstack().sort_values(ascending=False)[:5]

# Print the top 5 highest correlated features
print("HIGHEST CORRELATED PAIRS: \n", top_correlated_features, sep='')

# Top 10 correlation to determination status
print("\nHIGHEST CORRELATED TO DETERMINATION STATUS:\n", correlation_matrix['determination_status'].abs().sort_values(ascending=False)[1:11], sep='')
