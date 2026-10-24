from sklearn.cluster import KMeans 
import numpy as np
from sklearn.preprocessing import StandardScaler
# prepare the customer dataset 
# Each row contains: monthly spending and monthly visits
customers = np.array([
[1000, 2],
[1500, 3],
[2000, 2],
[5000, 6],
[6000, 7],
[5500, 5],
[12000, 12],
[15000, 14],
[13000, 11],
[3000, 4]])

# scale the spending and visits features
scaler = StandardScaler()
scaled_customers = scaler.fit_transform(customers)
# create the K-Means model with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42,n_init=10)
#Train the model and assign customers to clusters
clusters = kmeans.fit_predict(scaled_customers)
# Display the cluster assigned to each customer
print("CUSTOMER CLUSTER ASSIGNMENTS")
print("_ _ _ _ _ _ _ _ _ _ _ _ _ _ _")
for  customer_id, cluster in enumerate(clusters):
    print(f"Customer {customer_id + 1} belongs to Cluster {cluster}")
    #Display the number of customers in each cluster
    print("\nNUMBER OF CUSTOMERS IN EACH CLUSTER")
print("_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _")

for cluster in range(3):
    count = sum(clusters== cluster)
    print(f"Cluster {cluster}: {count} customers")
#_ _ _ _Bonus Task_ _ _ _
import matplotlib.pyplot as plt
# create a scatter plot with a different color for each cluster
plt.scatter(scaled_customers[:,0], scaled_customers[:, 1], c=clusters, cmap='viridis', edgecolors='k')
plt.xlabel('Scaled Monthly Spending')
plt.ylabel('Scaled Monthly Visits')
plt.title('Customer Segments (K-Means Clustering)')
plt.grid(True, linestyle='--', alpha=0.5)
#Display the plot window
plt.show()
