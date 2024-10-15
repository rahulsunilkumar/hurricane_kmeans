import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Streamlit app configuration
st.title("Hurricane Categorization using K-Means Clustering")
st.write("This app demonstrates categorization of hurricanes using K-means clustering based on Pressure and Size.")

# Define the dataset with given data points
data = pd.DataFrame({
    'Hurricane Name': [
        'Hurricane Katrina (2005)', 'Hurricane Harvey (2017)', 'Hurricane Ian (2022)',
        'Hurricane Andrew (1992)', 'Hurricane Irma (2017)', 'Hurricane Ivan (2004)',
        'Hurricane Charley (2004)'
    ],
    'Estimated Damage Cost (USD)': [160_000_000_000, 125_000_000_000, 60_000_000_000, 51_300_000_000, 50_000_000_000, 26_100_000_000, 14_000_000_000],
    'Wind Speed (mph)': [175, 130, 160, 175, 180, 165, 150],
    'Pressure (hPa)': [902, 938, 937, 922, 914, 910, 941]
})

# Define hurricane categories based on wind speed
def categorize_hurricane(wind_speed):
    if 74 <= wind_speed <= 95:
        return 1
    elif 96 <= wind_speed <= 110:
        return 2
    elif 111 <= wind_speed <= 130:
        return 3
    elif 131 <= wind_speed <= 155:
        return 4
    elif wind_speed > 155:
        return 5
    else:
        return np.nan

# Apply categorization based on wind speed
data['Actual Category'] = data['Wind Speed (mph)'].apply(categorize_hurricane)

# Display the data
st.subheader("Hurricane Data")
st.write(data)

# Prepare features for clustering
X = data[['Pressure (hPa)', 'Wind Speed (mph)']]

# Get user input for number of clusters (categories)
num_clusters = st.slider("Select number of clusters for K-Means", min_value=1, max_value=5, value=5)

# Perform K-Means Clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)

# Display the clustering results
st.subheader("Cluster Assignment")
st.write(data[['Hurricane Name', 'Pressure (hPa)', 'Wind Speed (mph)', 'Actual Category', 'Cluster']])

# Plotting the Clusters
st.subheader("Cluster Visualization")
fig, ax = plt.subplots()
scatter = ax.scatter(data['Pressure (hPa)'], data['Wind Speed (mph)'], c=data['Cluster'], cmap='viridis', alpha=0.6)
centroids = kmeans.cluster_centers_
ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=100, label='Centroids')
ax.set_xlabel('Pressure (hPa)')
ax.set_ylabel('Wind Speed (mph)')
ax.set_title('K-Means Clustering of Hurricanes')
ax.legend()
st.pyplot(fig)

# Evaluating the Results
st.subheader("Comparison of Clustering with Actual Categories")
st.write("The goal is to see if the clusters created by K-Means match the actual hurricane categories.")

# Show a simple comparison
st.write("Cluster vs. Actual Category")
st.write(data[['Hurricane Name', 'Actual Category', 'Cluster']])

# Calculate how many clusters match actual categories
matching_count = np.sum(data['Actual Category'] == data['Cluster'])
st.write(f"Number of hurricanes where cluster matches actual category: {matching_count} out of {len(data)}")
