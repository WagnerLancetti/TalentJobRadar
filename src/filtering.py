from src.clustering import create_binary_vector
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import silhouette_score
from collections import Counter
from scipy.stats import chi2_contingency
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import numpy as np

db_path = "./db/"

def read_data(filename, sep=';'): # Leitura do dataframe
    return pd.read_csv(filename, sep=sep)

def filter_data(df, selected_levels, selected_types_mapped): # Filtra onde nao menciona skill, e aplica os filtros do usuario
    indices_not_mention = df[df['soft_skills'].apply(lambda x: "Not mention" in x)].index
    df_filtered = df.drop(indices_not_mention)
    if selected_levels:
        df_filtered = df_filtered[df_filtered['seniority'].isin(selected_levels)]
    
    if selected_types_mapped:
        df_filtered = df_filtered[df_filtered['remote'].isin(selected_types_mapped)]
    df_filtered.reset_index(drop=True, inplace=True)
    return df_filtered

def preprocess_soft_skills(df_filtered): # Transformacao das habilidades em vetores binarios
    df_filtered['soft_skills'] = df_filtered['soft_skills'].apply(eval)
    mlb = MultiLabelBinarizer()
    skills_encoded = mlb.fit_transform(df_filtered['soft_skills'])
    skills_labels = mlb.classes_
    return skills_encoded, skills_labels

def calculate_coocurrence_matrix(skills_encoded): # Calcula a matriz de concorrencia, ou seja, quantas vezes as habilidades aparecem juntas
    return np.dot(skills_encoded.T, skills_encoded)

def cramers_v(confusion_matrix): # Calcula o valor da metrica V de cramer
    if confusion_matrix.shape[0] == 1 or confusion_matrix.shape[1] == 1:
        return 0
    chi2, _, _, _ = chi2_contingency(confusion_matrix, correction=False)
    n = confusion_matrix.sum()
    phi2 = chi2 / n if n > 0 else 0
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1)) if n > 1 else 0
    rcorr = r - ((r-1)**2)/(n-1) if n > 1 else 1
    kcorr = k - ((k-1)**2)/(n-1) if n > 1 else 1
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1))) if min((kcorr-1), (rcorr-1)) > 0 else 0

def calculate_assoc_matrix(coocurrence_matrix, num_skills): # Criacao da matriz de associacao, usando a matriz de concorrencia e o v de cramer
    assoc_matrix = np.zeros((num_skills, num_skills))
    for i in range(num_skills):
        for j in range(num_skills):
            if i != j: # Se os valores sao diferentes, transforma em matriz as linhas para calcular o V de cramer entre os dados
                confusion_matrix = np.array([
                    [coocurrence_matrix[i, i], coocurrence_matrix[i, j]],
                    [coocurrence_matrix[j, i], coocurrence_matrix[j, j]]
                ])
                assoc_matrix[i, j] = cramers_v(confusion_matrix)
            else:
                assoc_matrix[i, j] = 1
    return assoc_matrix

def calculate_silhouette_score(X, Z, n_clusters): # Calcula o valor de Silhouette Score dos clusters
    labels = hierarchy.fcluster(Z, n_clusters, criterion='maxclust')
    if len(np.unique(labels)) < 2:
        return -1  # Retorna -1 se nao ha clusters suficientes para calcular o Silhouette Score
    return silhouette_score(X, labels)

def perform_clustering(assoc_df, n_clusters_range): # Encontra a melhor configuracao dos clusters dado o valor de Silhouette Score
    X = assoc_df.values
    Z = hierarchy.linkage(X, method='ward')
    
    best_n_clusters = None
    best_score = -1
    
    for n_clusters in n_clusters_range: # Calcula os valores de todos os clusters para encontrar a melhor configuracao
        score = calculate_silhouette_score(X, Z, n_clusters)
        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters
    
    return best_n_clusters, Z

def create_cluster_map(labels, skills_labels, best_n_clusters): # Cria o mapeamento do cluster, habilidades e de qual cluster elas fazem parte
    cluster_map = {i: [] for i in range(1, best_n_clusters + 1)}
    for skill, label in zip(skills_labels, labels):
        cluster_map[label].append(skill)
    return cluster_map

def convert_to_dataframe(cluster_map): # Converter o cluster em um DF
    cluster_list = []
    for cluster_id, skills in cluster_map.items():
        for skill in skills:
            cluster_list.append({'Cluster': cluster_id, 'Skill': skill})
    return pd.DataFrame(cluster_list)

def recover_cluster_map(clusters_df): # Cria o mapeamento dos clusters e das habilidades
    return clusters_df.groupby('Cluster')['Skill'].apply(list).to_dict()

def percent_clusters(df, clusters): # Calcula quantas vezes as habilidades, presentes nos clusters, aparecem nos dados
    jobs = df['soft_skills']

    total_jobs = len(jobs)

    habilidades = [habilidade for job in jobs for habilidade in job] # Conta a ocorrencia das habilidades
    contagem_habilidades = Counter(habilidades)

    # Calcula o percentual
    percentuais = {habilidade: round((contagem / total_jobs) * 100, 2) for habilidade, contagem in contagem_habilidades.items()}

    percentuais_clusters = {}
    for cluster_id, habilidades in clusters.items(): # Mapeia para cada cluster a porcentagem que as habilidades aparecem
        percentuais_clusters[cluster_id] = {habilidade: percentuais.get(habilidade, 0) for habilidade in habilidades}

    return percentuais_clusters

def create_uniques(clusters): # Encontra quais sao as habilidades da base de dados e o mapeamento dela no vetor binario em ordem alfabetica
    uniques = []
    for cluster_id, skills in clusters.items():
        for elem in skills:
            uniques.append(elem)
    uniques.append('Not mention')
    uniques = sorted(uniques)
    map_positions = {item: idx for idx, item in enumerate(uniques)} # Cria o mapeamento: 0 = Adaptable, 1 = Assertive, etc
    vector_size = len(uniques)
    binary_vectors = {cluster: create_binary_vector(skills, map_positions, vector_size) for cluster, skills in clusters.items()}
    return uniques, map_positions, binary_vectors

def create_clusters(selected_levels, selected_types_mapped): # Cria os clusters baseado nos filtros que o usuario definiu
    df = read_data(db_path+'LinkedInPt.csv', sep = ";")
    df_filtered = filter_data(df, selected_levels, selected_types_mapped)
    skills_encoded, skills_labels = preprocess_soft_skills(df_filtered)
    coocurrence_matrix = calculate_coocurrence_matrix(skills_encoded)
    assoc_matrix = calculate_assoc_matrix(coocurrence_matrix, len(skills_labels))
    assoc_df = pd.DataFrame(assoc_matrix, index=skills_labels, columns=skills_labels)
    n_clusters_range = range(4, 11)
    best_n_clusters, Z = perform_clustering(assoc_df, n_clusters_range)
    labels = hierarchy.fcluster(Z, best_n_clusters, criterion='maxclust')
    cluster_map = create_cluster_map(labels, skills_labels, best_n_clusters)
    clusters_df = convert_to_dataframe(cluster_map)
    recovered_cluster_map = recover_cluster_map(clusters_df)
    percentuais = percent_clusters(df_filtered,recovered_cluster_map)
    return recovered_cluster_map, percentuais