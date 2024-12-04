import importlib
import utils
import metrics
import es
import prediction_fn
import chatgpt

importlib.reload(utils)
importlib.reload(metrics)
importlib.reload(es)
importlib.reload(prediction_fn)
importlib.reload(chatgpt)

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import os
from utils import display_images_in_row, display_image
from es import create_client, get_clip_cn_embedding
from scipy.spatial.distance import cdist
import argparse

def centroid_and_furthest_images(embeddings_scaled, labels, centroids, user_df, image_dir):
    max_distances = []
    furthest_images = []
    centroid_images = []

    for cluster_id in range(len(centroids)):
        cluster_points = embeddings_scaled[labels == cluster_id]

        distances = cdist(cluster_points, [centroids[cluster_id]], metric='euclidean')

        max_distance_index = np.argmax(distances)
        max_distance = distances[max_distance_index][0]

        min_distance_index = np.argmin(distances)

        furthest_docid = user_df[user_df['cluster'] == cluster_id]['docid'].iloc[max_distance_index]
        furthest_image_path = f"{image_dir}/{str(furthest_docid)}.jpg"

        centroid_docid = user_df[user_df['cluster'] == cluster_id]['docid'].iloc[min_distance_index]
        centroid_image_path = f"{image_dir}/{str(centroid_docid)}.jpg"

        max_distances.append(max_distance)
        furthest_images.append(furthest_image_path)
        centroid_images.append(centroid_image_path)

    return max_distances, furthest_images, centroid_images, centroids

def fetch_embedding(docid, es_cli):
    embedding = get_clip_cn_embedding(es_cli, docid)
    if embedding is None:
        return []
    return embedding

def cluster_user_embeddings(user_df, es_cli, n_clusters):
    embeddings_list = np.array([fetch_embedding(docid, es_cli) for docid in user_df['docid']])
    embeddings_list = embeddings_list[~np.isnan(embeddings_list).any(axis=1)]

    if embeddings_list.shape[0] == 0:
        return np.array([]), np.array([]), np.array([])

    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings_list)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings_scaled)
    return kmeans.labels_, embeddings_scaled, kmeans.cluster_centers_

def plot_clusters(embeddings, labels, title):
    if embeddings.shape[0] == 0:
        print(f'No embeddings to plot for {title}')
        return

    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    df_plot = pd.DataFrame(reduced_embeddings, columns=['Component 1', 'Component 2'])
    df_plot['Cluster'] = labels

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='Component 1', y='Component 2', hue='Cluster', palette='viridis', data=df_plot, marker='o')
    plt.title(title)
    plt.show()

def main(data_file, image_dir, output_file):
    es_cli = create_client()

    df = pd.read_csv(data_file)

    df['embedding'] = df['docid'].apply(lambda docid: fetch_embedding(docid, es_cli))
    df = df[df['embedding'].apply(lambda x: len(x) > 0)] 

    clusters_by_users = [5, 5, 5, 5, 5, 5, 5, 5]
    idx = 0

    csv_data = []

    for user_id, user_df in df.groupby('uid'):
        print(f'----User {user_id}----')
        labels, user_embeddings, centroids = cluster_user_embeddings(user_df, es_cli, n_clusters=clusters_by_users[idx])

        if len(labels) > 0:
            user_df['cluster'] = labels

            max_distances, furthest_images, centroid_images, centroids = centroid_and_furthest_images(
                user_embeddings, labels, centroids, user_df, image_dir
            )

            for cluster_id, max_dist in enumerate(max_distances):
                print(f'User {user_id}, Cluster {cluster_id}, Max Distance to Centroid: {max_dist:.4f}')

                print(f'Image furthest from centroid for Cluster {cluster_id}:')
                display_image(furthest_images[cluster_id])

                print(f'Centroid image for Cluster {cluster_id}:')
                display_image(centroid_images[cluster_id])

                csv_data.append({
                    'userid': user_id,
                    'clusterid': cluster_id,
                    'centroid_vector': centroids[cluster_id],
                    'centroid_docid': user_df[user_df['cluster'] == cluster_id]['docid'].iloc[np.argmin(
                        cdist([centroids[cluster_id]], user_embeddings[labels == cluster_id], metric='euclidean'))],
                    'threshold': max_dist
                })

            plot_clusters(user_embeddings, labels, f'User {user_id} Clustering')

            for cluster_id in np.unique(labels):
                cluster_images = user_df[user_df['cluster'] == cluster_id]['docid'].tolist()
                cluster_images = [f"{str(docid)}.jpg" for docid in cluster_images]
                if len(cluster_images) > 0:
                    print(f'User {user_id}, Cluster {cluster_id}')
                    display_images_in_row(cluster_images[:10]) 
        idx += 1

    csv_df = pd.DataFrame(csv_data)
    csv_df.to_csv(output_file, index=False)
    print(f'Data saved to {output_file}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster user embeddings and generate results.")
    parser.add_argument('--data_file', required=True, help='Path to the input CSV file containing user data')
    parser.add_argument('--image_dir', required=True, help='Path to the directory containing image files')
    parser.add_argument('--output_file', required=True, help='Path to the output CSV file for saving results')

    args = parser.parse_args()

    main(args.data_file, args.image_dir, args.output_file)