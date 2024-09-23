import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV file
csv_file = 'your_data.csv'
data = pd.read_csv(csv_file)

# Encode categorical columns
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Scale data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.select_dtypes(include=[float, int]))

# Apply GMM Clustering
gmm = GaussianMixture(n_components=3, random_state=42)
gmm_labels = gmm.fit_predict(data_scaled)
data['GMM_Cluster'] = gmm_labels

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(data_scaled)
data['KMeans_Cluster'] = kmeans_labels

# Calculate Silhouette Scores
gmm_silhouette = silhouette_score(data_scaled, gmm_labels)
kmeans_silhouette = silhouette_score(data_scaled, kmeans_labels)
print(f'Silhouette Score for GMM: {gmm_silhouette:.2f}')
print(f'Silhouette Score for K-means: {kmeans_silhouette:.2f}')

# Visualization for 2D Data
if data_scaled.shape[1] == 2:
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=data_scaled[:, 0], y=data_scaled[:, 1], hue=gmm_labels, palette='viridis')
    plt.title('GMM Clustering')
    
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=data_scaled[:, 0], y=data_scaled[:, 1], hue=kmeans_labels, palette='viridis')
    plt.title('K-means Clustering')
    plt.show()

# Visualization for 3D Data
elif data_scaled.shape[1] == 3:
    fig = plt.figure(figsize=(14, 6))
    
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(data_scaled[:, 0], data_scaled[:, 1], data_scaled[:, 2], c=gmm_labels, cmap='viridis')
    ax.set_title('GMM Clustering')
    
    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(data_scaled[:, 0], data_scaled[:, 1], data_scaled[:, 2], c=kmeans_labels, cmap='viridis')
    ax.set_title('K-means Clustering')
    
    plt.show()

else:
    print('Data is not 2D or 3D, skipping visualization.')

# Save clustered data to a CSV file
output_csv_file = 'clustered_data.csv'
data.to_csv(output_csv_file, index=False)
print(f'Clustered data saved to {output_csv_file}')
