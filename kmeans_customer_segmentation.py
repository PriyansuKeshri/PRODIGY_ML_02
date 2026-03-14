import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# Load dataset
data = pd.read_csv("Mall_Customers.csv")

# Preview dataset
print("Dataset Preview:")
print(data.head())


# Select features for clustering
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]


# Determine optimal number of clusters using the Elbow Method
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


# Plot the Elbow Graph
plt.figure()
plt.plot(range(1, 11), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()


# Train K-Means with optimal clusters (usually 5 for this dataset)
kmeans = KMeans(n_clusters=5, random_state=42)

# Fit model
y_kmeans = kmeans.fit_predict(X)


# Plot clusters
plt.figure()

plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_kmeans)

# Plot cluster centroids
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200,
    marker='X'
)

plt.title("Customer Segments")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()