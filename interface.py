import pandas as pd
import numpy as np
import zipfile
import ast
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
from collections import Counter
import streamlit as st
from pyngrok import ngrok


db_path = "./db/"

def import_clusters(): # Extração dos clusters
    with zipfile.ZipFile(db_path+"clusters.zip", 'r') as zipf:
        zipf.extractall()
    recovered_clusters_df = pd.read_csv('clusters.csv')
    recovered_cluster_map = recovered_clusters_df.groupby('Cluster')['Skill'].apply(list).to_dict()
    return recovered_cluster_map

def percent_clusters(df, clusters):
    jobs = df['soft_skills'].apply(eval)

    total_jobs = len(jobs)

    habilidades = [habilidade for job in jobs for habilidade in job]
    contagem_habilidades = Counter(habilidades)

    percentuais = {habilidade: round((contagem / total_jobs) * 100, 2) for habilidade, contagem in contagem_habilidades.items()}

    percentuais_clusters = {}
    for cluster_id, habilidades in clusters.items():
        percentuais_clusters[cluster_id] = {habilidade: percentuais.get(habilidade, 0) for habilidade in habilidades}

    return percentuais_clusters

def import_similarity_matrix(): # Extração da matrix
    with zipfile.ZipFile(db_path+"matrix_similarity.zip", 'r') as zipf:
        zipf.extractall()
    recovered_matrix_similarity = np.loadtxt('matrix_similarity.csv', delimiter=',')
    return recovered_matrix_similarity

def create_binary_vector(skills, map_positions, vector_size):
    vector = [0] * vector_size
    for skill in skills:
        position = map_positions[skill]
        vector[position] = 1
    return vector

def map_vector(skills, map_positions, total_skills):
    bin_vector = [0] * total_skills
    for skill in skills:
        if (skill == 'Not mention'):
            bin_vector[map_positions[skill]] = 0
        elif skill in map_positions:
            bin_vector[map_positions[skill]] = 1
    return bin_vector

def recommender_job(vector, matrix):
    result_cos = cosine_similarity(vector, matrix)
    return result_cos

def jaccard_similarity(vetor_a, vetor_b):
    return jaccard_score(vetor_a, vetor_b, average='binary')

def ranking_skills(percentuais_clusters):
    soma_habilidades = {}

    for subdict in percentuais_clusters.values():
        for habilidade, valor in subdict.items():
            if habilidade in soma_habilidades:
                soma_habilidades[habilidade] += valor
            else:
                soma_habilidades[habilidade] = valor

    ranking = sorted(soma_habilidades.items(), key=lambda x: x[1], reverse=True)

    return ranking

def RecomenderJob(df, vector, similarity_matrix, map_positions, uniques):
    binary_vector = map_vector(vector, map_positions, len(uniques)) #  Vetor que será recebido da aplicação

    norm_matrix = norm(similarity_matrix, axis=1)
    norm_vector = norm(binary_vector)

    # evitar divisão por zero
    epsilon = 1e-10
    norm_matrix = np.where(norm_matrix == 0, epsilon, norm_matrix)
    norm_vector = epsilon if norm_vector == 0 else norm_vector

    result = np.dot(similarity_matrix, binary_vector) / (norm_matrix * norm_vector)

    index = np.argsort(result)[::-1]
    similarity_rank = result[index]

    # Remove os resultados nan
    similarity_rank = similarity_rank[~np.isnan(similarity_rank)]

    sorted_indices = index[~np.isnan(result[index])]
    indices = []
    for i, (idx, sim) in enumerate(zip(sorted_indices, similarity_rank)):
        if i >= 5:
            break
        indices.append(idx)
    selected_rows = df.loc[indices]
    return selected_rows


def RecomenderSkill(vector, uniques, map_positions, binary_vectors, percentuais_clusters):
    binary_vector = map_vector(vector, map_positions, len(uniques))
    similaridades = []

    for elem in binary_vectors.values():
        similaridades.append(jaccard_similarity(elem, binary_vector))

    # print(f"Similaridades: {similaridades}")
    ranking_indices = np.argsort(similaridades)[::-1]

    cluster_ranking = [list(binary_vectors.keys())[i] for i in ranking_indices]
    # print(f"Ranking dos clusters (do maior para o menor): {cluster_ranking}")
    id = cluster_ranking[0]
    resultado_cluster_1 = percentuais_clusters[id]

    habilidades_filtradas = {k: v for k, v in resultado_cluster_1.items() if k not in vector}

    habilidades_ordenadas = dict(sorted(habilidades_filtradas.items(), key=lambda item: item[1], reverse=True))
    
    rank_skills = ranking_skills(percentuais_clusters)
    rank_filtrado = {k: v for k, v in rank_skills if k not in vector}

    return habilidades_ordenadas, rank_filtrado

def create_card(job_title, company, skills, link):
    skills = ast.literal_eval(skills)
    skills_formatted = ', '.join(skills)  # Formatar a lista de habilidades como uma string
    card_html = f"""
    <div style='border: 1px solid #ddd; border-radius: 10px; padding: 16px; margin-bottom: 16px; 
                box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.1); width: 100%; display: inline-block;'>
        <h3>{job_title}</h3>
        <p><strong>Empresa:</strong> {company}</p>
        <p><strong>Habilidades:</strong> {skills_formatted}</p>
        <a href='{link}' target='_blank' style='text-decoration: none; color: #1a73e8;'>{link}</a>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def create_podium(habilidades_show, rank_skills):
    st.markdown("Recomendações que podem melhorar suas possibilidades de emprego:")
    st.markdown("<br>", unsafe_allow_html=True)  # Adiciona uma linha em branco

    css = """
    <style>
    .podium {
        display: flex;
        justify-content: space-around;
        align-items: center;
        margin-bottom: 20px;
    }

    .first {
        font-size: 24px;
        font-weight: bold;
        color: gold;
        text-align: center;
        flex: 1;
    }

    .second {
        font-size: 20px;
        color: silver;
        text-align: center;
        flex: 0.5;
    }

    .third {
        font-size: 16px;
        color: #cd7f32;
        text-align: center;
        flex: 0.5;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

    # Determinar quantas habilidades serão exibidas
    num_habilidades = len(habilidades_show)
    if num_habilidades > 0:
        for i, habilidade in enumerate(habilidades_show, 1):
            if i == 1:
                st.markdown(f"<span class='first'>{i}º - {habilidade}</span>", unsafe_allow_html=True)
            elif i == 2:
                st.markdown(f"<span class='second'>{i}º - {habilidade}</span>", unsafe_allow_html=True)
            elif i == 3:
                st.markdown(f"<span class='third'>{i}º - {habilidade}</span>", unsafe_allow_html=True)
    elif len(rank_skills) > 0:
        for i, (habilidade, valor) in enumerate(rank_skills.items(), 1):
            if i == 1:
                st.markdown(f"<span class='first'>{i}º - {habilidade}</span>", unsafe_allow_html=True)
            elif i == 2:
                st.markdown(f"<span class='second'>{i}º - {habilidade}</span>", unsafe_allow_html=True)
            elif i == 3:
                st.markdown(f"<span class='third'>{i}º - {habilidade}</span>", unsafe_allow_html=True)
    else:
        st.write("Você posssui todas as soft skills!!")

df = pd.read_csv(db_path+"LinkedInPtFilter_atualizado.csv", sep = ";")
jobs = df['soft_skills'].apply(eval)

clusters = import_clusters()
similarity_matrix = import_similarity_matrix()
percentuais_clusters = percent_clusters(df, clusters)

uniques = []
for cluster_id, skills in clusters.items():
    for elem in skills:
        uniques.append(elem)
uniques.append('Not mention')
uniques = sorted(uniques)
map_positions = {item: idx for idx, item in enumerate(uniques)}
vector_size = len(uniques)
binary_vectors = {cluster: create_binary_vector(skills, map_positions, vector_size) for cluster, skills in clusters.items()}

# _---------------INTERFACE----------------_ #

st.title('Recomendação - Soft Skills')

# Texto informativo
st.write('Recomendação de empregos baseados nas suas habilidades, ou desenvolvimento de habilidades para alavancar suas possibilidades!')

# Entrada de Texto
options = [f'{skill}' for skill in uniques]

habilidades_selecionadas = st.multiselect('Selecione suas habilidades:', options)

# Slider
faixa_salarial = st.slider('Pretensão Salarial:', 1000, 10000, 5000, step=100)

# Botão
col1, col2 = st.columns([1, 1])

botao_empregos_pressed = False
botao_habilidades_pressed = False

# Adicionar um botão em cada coluna do meio
with col1:
    if st.button('Buscar Empregos'):
        result = RecomenderJob(df, habilidades_selecionadas, similarity_matrix, map_positions, uniques)
        link_address = "https://www.linkedin.com/jobs/view/"
        lista = []
        for index, row in result.iterrows():
            lista.append([row['title'], row['org_name'], row['soft_skills'], f"{link_address}{row['ID']}"])
        botao_empregos_pressed = True

with col2:
    if st.button('Melhorar Habilidades'):
        habilidades_result, rank_skills = RecomenderSkill(habilidades_selecionadas, uniques, map_positions, binary_vectors, percentuais_clusters)
        habilidades_show = []
        for i, habilidade in enumerate(habilidades_result, 1):
            if i > 3:
                break
            habilidades_show.append(habilidade)
        botao_habilidades_pressed = True

if botao_empregos_pressed:
    botao_habilidades_pressed = False
    st.markdown("---")
    st.title("Resultados:")
    for item in lista:
        create_card(item[0], item[1], item[2], item[3])

if botao_habilidades_pressed:
    botao_empregos_pressed = False
    st.markdown("---")
    st.title("Resultados:")
    create_podium(habilidades_show, rank_skills)