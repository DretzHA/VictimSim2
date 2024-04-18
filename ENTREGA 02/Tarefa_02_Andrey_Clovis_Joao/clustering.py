import numpy as np


def k_means_clustering(victims_features, k=4):
    """K-Means algorithm
    @:param victims_features: list of labels and features by victim"""

    # Definição dos centroides iniciais
    indexes = np.random.choice(list(range(0, len(victims_features))), k, replace=False)
    centroids = np.array([victims_features[i][1] for i in indexes])

    iteration = 0
    while True:
        iteration += 1
        stop = True

        # 'Rotulagem dos dados'
        victim_number = 0
        distances_to_centroids = []
        for victim_data in victims_features:
            distance_per_victim = []
            for centroid in centroids:
                distance_per_victim.append(np.linalg.norm(victim_data[1] - centroid))
            distance_per_victim = np.array(distance_per_victim)
            cluster = np.argmin(distance_per_victim)
            distances_to_centroids.append(distance_per_victim)
            if cluster != victims_features[victim_number][0]:
                victims_features[victim_number][0] = cluster
                stop = False
            victim_number += 1

        # 'Cálculo dos novos centroides'
        for cluster in range(0, k):
            victims_in_cluster = list(filter(lambda x: x[0] == cluster, victims_features))
            victims_in_cluster = [i[1] for i in victims_in_cluster]
            centroids[cluster] = np.mean(victims_in_cluster, axis=0)

        if stop and iteration > 1:
            break

    # Cálculo da inércia
    mins = np.min(distances_to_centroids, axis=1)
    inertia = np.sum(np.square(mins))

    return inertia, victims_features