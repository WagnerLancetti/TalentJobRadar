import pandas as pd
import zipfile
import numpy as np
import os

db_path = "./db/"

def import_clusters(): # Extração dos clusters
    with zipfile.ZipFile(db_path+"clusters.zip", 'r') as zipf:
        zipf.extractall()
    recovered_clusters_df = pd.read_csv('clusters.csv')
    recovered_cluster_map = recovered_clusters_df.groupby('Cluster')['Skill'].apply(list).to_dict()
    return recovered_cluster_map

def import_similarity_matrix(): # Extração da matrix
    with zipfile.ZipFile(db_path+"matrix_similarity.zip", 'r') as zipf:
        zipf.extractall()
    recovered_matrix_similarity = np.loadtxt('matrix_similarity.csv', delimiter=',')
    recovered_matrix_similarity = np.round(recovered_matrix_similarity).astype(int)
    return recovered_matrix_similarity

def create_binary_vector(skills, map_positions, vector_size):
    vector = [0] * vector_size
    for skill in skills:
        position = map_positions[skill]
        vector[position] = 1
    return vector