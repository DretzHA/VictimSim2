import numpy as np
from sklearn.metrics import silhouette_score
import plotly.express as px

def k_means_scikit(victims):

        from sklearn.cluster import KMeans
        print('K-means through scikit learn')
        victims_features = []
        for victim in victims.values():
            victims_features.append(np.array(list(victim[0])))
        k_means = KMeans(n_clusters=4)
        labels = k_means.fit_predict(victims_features)
        print(f'Inertia scikit: {k_means.inertia_}')
        print(f'Silhouette score: {silhouette_score(np.array(victims_features), labels)}')
        x = [i[0] for i in victims_features]
        y = [i[1] for i in victims_features]
        scatter = px.scatter(x=x, y=y, color=labels)
        scatter.update_traces(showlegend=False)
        scatter.update_yaxes(autorange='reversed')
        scatter.write_image('./scikit_k_means_clustering.png', format='png', width=500, height=500)

def agg_clustering(victims):

        from sklearn.cluster import AgglomerativeClustering
        print('Agglomerative clustering through scikit learn')
        victims_features = []
        for victim in victims.values():
            victims_features.append(np.array(list(victim[0])))
        agg_clustering = AgglomerativeClustering(n_clusters=4)
        labels = agg_clustering.fit_predict(victims_features)
        x = [i[0] for i in victims_features]
        y = [i[1] for i in victims_features]
        print(f'Silhouette score: {silhouette_score(np.array(victims_features), labels)}')
        scatter = px.scatter(x=x, y=y, color=labels)
        scatter.update_traces(showlegend=False)
        scatter.update_yaxes(autorange='reversed')
        scatter.write_image('./scikit_agg_clustering.png', format='png', width=500, height=500)