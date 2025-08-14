# K-Means Clustering - Customer Segmentation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Load dataset
df = pd.read_csv('Mall_Customers.csv')  # Dataset file from Kaggle
print("First 5 rows of dataset:\n", df.head())

# Step 2: Select features for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Step 3: Feature Scaling (optional but good practice)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Find the optimal number of clusters using Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig('elbow_plot.png')
plt.show()

# Step 5: Train KMeans model with optimal clusters (assume 5 from elbow)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Step 6: Add cluster info to dataset
df['Cluster'] = y_kmeans

# Step 7: Visualize clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', 
                data=df, palette='Set1', s=100)
plt.title('Customer Segments')
plt.savefig('kmeans_clusters.png')
plt.show()
