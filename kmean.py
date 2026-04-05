import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


data = {
    'CustomerID': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
    'Annual_Income': [15,16,17,18,19,60,62,63,65,68,70,72,75,78,80],
    'Spending_Score': [39,81,6,77,40,55,56,52,59,61,65,68,70,72,75]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)


X = df[['Annual_Income', 'Spending_Score']]


wcss = []

for i in range(1, 6):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 6), wcss)
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()


kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

print("\nClustered Data:")
print(df)

plt.scatter(df['Annual_Income'], df['Spending_Score'], c=df['Cluster'])
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using K-Means")
plt.show()

print("\nCluster Centers:")
print(kmeans.cluster_centers_)
