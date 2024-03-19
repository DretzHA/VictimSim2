from sklearn.cluster import KMeans
import numpy as np

#cluster e divisão de vitimas por agentes de resgate
def victims_clustering(merged_victims):
    v_array =np.empty((0,6), float) #array com coordenadas e sinais vitais
    cluster_array = np.empty((0,8), float) #array para features do cluster
    seq_array = np.empty((0,1), int)
    resc1_victims = {} #dictionaty de vitimas para resgate 1
    resc2_victims = {} #dictionaty de vitimas para resgate 2
    resc3_victims = {} #dictionaty de vitimas para resgate 3
    resc4_victims = {} #dictionaty de vitimas para resgate 4
    for seq, data in merged_victims.items():
        coord, vital_signals = data
        x, y = coord
        coord_array = np.array([[x, y]]) #transforma x,y num array
        v_array = np.append(coord_array,vital_signals) #junta array de coordenadas com sinais vitais
        cluster_array = np.vstack([cluster_array, v_array]) #passa valores para cluster array
        seq_array = np.vstack([seq_array, seq])
    if (len(cluster_array)) <= 4:
        num_clusters = len(cluster_array) #se menor que 4 vitimas achadas, então numero de cluster = vitimas
    else:
        num_clusters = 4 #se mais de 4 vitimas achadas, então numéro de clusters = 4 (total de resgates)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(cluster_array) #clusterização por k-means - 
                                                                                               #pode-se cria outro array com diferentes colunas(features) para uso no k-mean


    #logica para separar vitimas do cluster por agente de resgate
    for i in range(0,len(merged_victims)):
        if (kmeans.labels_[i])==0:
            resc1_victims = {key:merged_victims[key] for key in [seq_array[i][0]]}
        elif (kmeans.labels_[i])==1:
            resc2_victims = {key:merged_victims[key] for key in [seq_array[i][0]]}
        elif (kmeans.labels_[i])==2:
            resc3_victims = {key:merged_victims[key] for key in [seq_array[i][0]]}
        else:
            resc4_victims = {key:merged_victims[key] for key in [seq_array[i][0]]}
            
    
    for seq, data in resc1_victims.items():
         coord, vital_signals = data
         x, y = coord
         print(f"RESGATE1: Victim seq number: {seq} at ({x}, {y}) vs: {vital_signals}")
         
    for seq, data in resc2_victims.items():
         coord, vital_signals = data
         x, y = coord
         print(f"RESGATE2: Victim seq number: {seq} at ({x}, {y}) vs: {vital_signals}")
    
    for seq, data in resc3_victims.items():
         coord, vital_signals = data
         x, y = coord
         print(f"RESGATE3: Victim seq number: {seq} at ({x}, {y}) vs: {vital_signals}")
         
    for seq, data in resc4_victims.items():
         coord, vital_signals = data
         x, y = coord
         print(f"RESGATE4: Victim seq number: {seq} at ({x}, {y}) vs: {vital_signals}")