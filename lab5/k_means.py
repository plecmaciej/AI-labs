import numpy as np
import random

def initialize_centroids_forgy(data, k):
    # TODO implement random initialization
    centroids = []
    #for _ in range(1,k):
    centroids = random.choices((data), k = k)
    return centroids

def initialize_centroids_kmeans_pp(data, k):
    # TODO implement kmeans++ initizalization
    centroids = []
    centroids.append(random.choice(data))
    for _ in range(1, k):
        distances = []
        for point in data:

            min_distance = min(np.linalg.norm(point - centroid) for centroid in centroids)
            distances.append(min_distance)


        max_distance_index = distances.index(max(distances))
        centroids.append(data[max_distance_index])

    return centroids

def assign_to_cluster(data, centroids):
    assigned_clusters = []


    for point in data:

        distances = [np.linalg.norm(point - centroid) for centroid in centroids]

        closest_cluster_index = np.argmin(distances)

        assigned_clusters.append(closest_cluster_index)

    return assigned_clusters

def update_centroids(data, assignments):

    new_centroids = []


    for centroid in np.unique(assignments):

        cluster_points = data[assignments == centroid]

        new_centroid = np.mean(cluster_points, axis=0)
        # Dodajemy nowy centroid do listy
        new_centroids.append(new_centroid)

    return (new_centroids)

def mean_intra_distance(data, assignments, centroids):
    assignments = np.array(assignments)  # Konwersja listy na tablicę numpy
    return np.sqrt(np.sum((data - np.array(centroids)[assignments])**2))

def k_means(data, num_centroids, kmeansplusplus= False):
    # centroids initizalization
    if kmeansplusplus:
        centroids = initialize_centroids_kmeans_pp(data, num_centroids)
    else: 
        centroids = initialize_centroids_forgy(data, num_centroids)

    
    assignments  = assign_to_cluster(data, centroids)
    for i in range(100): # max number of iteration = 100
        print(f"Intra distance after {i} iterations: {mean_intra_distance(data, assignments, centroids)}")
        centroids = update_centroids(data, assignments)
        new_assignments = assign_to_cluster(data, centroids)
        if np.all(new_assignments == assignments): # stop if nothing changed
            break
        else:
            assignments = new_assignments

    return new_assignments, centroids, mean_intra_distance(data, new_assignments, centroids)         

