import pandas as pd
import numpy as np
import zipfile
import ast
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
from collections import Counter
import streamlit as st
import base64
from pyngrok import ngrok
from PIL import Image


db_path = "./db/"

def get_image_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

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


def jaccard_similarity(A, B):
    A = np.array(A, dtype=np.int32)  # Converter A para array numérico
    B = np.array(B, dtype=np.int32)  # Converter B para array numérico

    intersection = np.sum(np.logical_and(A, B))
    union = np.sum(np.logical_or(A, B))
    
    return intersection / union if union > 0 else 0  # Evita divisão por zero

def increase_similarity(A, B, map_positions):
    old_similarity = jaccard_similarity(A, B)
    new_similarity = old_similarity
    ids = []
    for i in range(len(A)):
        if B[i] == 1 and A[i] == 0:
            A[i] = 1  # adiciona a habilidade em A
            ids.append(i)
            new_similarity = jaccard_similarity(A, B)
    str_return = ""
    if new_similarity > old_similarity:
        str_return = f"É recomendado trabalhar as seguintes habilidades para melhorar sua aptidão para {new_similarity*100:.2f}% nesse emprego: "
        added_skills = [skill for skill, idx in map_positions.items() if idx in ids]
        str_return += ", ".join(added_skills)
    else:
        str_return = "Você já está o melhor preparado possível para esse emprego!"
    return str_return

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
    return selected_rows, similarity_rank[:len(indices)]


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


def get_base64_of_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def create_card(card_id, job_title, company, skills, link, similarity, vector, map_positions, uniques):
    binary_vector = map_vector(vector, map_positions, len(uniques))
    skills_list = ast.literal_eval(skills)
    binary_vector2 = map_vector(skills_list, map_positions, len(uniques))
    skill = ast.literal_eval(skills)
    skills_formatted = ', '.join(skill)  # Formatar a lista de habilidades como uma string
    image_path = "./imgs/upgrade-nobg.png"
    img_base64 = get_base64_of_image(image_path)
    num = f"{similarity*100:.2f}%"

    result = increase_similarity(binary_vector, binary_vector2, map_positions)

    # HTML do card
    if result and result != "Você já está o melhor preparado possível para esse emprego!":
        result_html = f"<p style='color: #00ff95; margin-top: 25px;'>{result}</p>"
    else:
        result_html = f"<p style='margin-top: 0px;'></p>"

    # HTML do card
    card_html = f"""
    <div id="card_{card_id}" style='border: 1px solid #ddd; border-radius: 10px; padding: 16px; margin-bottom: 16px; 
                box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.1); width: 100%; display: flex; flex-direction: column;'>
        <div style='flex: 1;'>
            <h3>{job_title}</h3>
            <p><strong>Empresa:</strong> {company}</p>
            <p><strong>Habilidades:</strong> {skills_formatted}</p>
            <div style='display: flex; justify-content: space-between; align-items: center;'>
                <a href='{link}' target='_blank' style='text-decoration: none; color: #1a73e8;'>Saiba mais</a>
                <span style='margin-left: auto; margin-right: 50px; font-size: 16px;'>Aptidão: {num}</span>
            </div>
        <div style='display: flex; flex-direction: column;'>
            <div style='margin-top: auto;'>
                {result_html}
            </div>
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def create_podium(habilidades_show, rank_skills, map_translate):
    st.markdown("Recomendações de habilidades que podem melhorar suas possibilidades de emprego:")
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
            habilidade_traduzida = map_translate.get(habilidade, habilidade)  # Obtém a tradução ou usa o original se não houver tradução
            if i == 1:
                st.markdown(f"<span class='first'>{i}º - {habilidade_traduzida}</span>", unsafe_allow_html=True)
            elif i == 2:
                st.markdown(f"<span class='second'>{i}º - {habilidade_traduzida}</span>", unsafe_allow_html=True)
            elif i == 3:
                st.markdown(f"<span class='third'>{i}º - {habilidade_traduzida}</span>", unsafe_allow_html=True)
    elif len(rank_skills) > 0:
        for i, (habilidade, valor) in enumerate(rank_skills.items(), 1):
            habilidade_traduzida = map_translate.get(habilidade, habilidade)  # Obtém a tradução ou usa o original se não houver tradução
            if i == 1:
                st.markdown(f"<span class='first'>{i}º - {habilidade_traduzida}</span>", unsafe_allow_html=True)
            elif i == 2:
                st.markdown(f"<span class='second'>{i}º - {habilidade_traduzida}</span>", unsafe_allow_html=True)
            elif i == 3:
                st.markdown(f"<span class='third'>{i}º - {habilidade_traduzida}</span>", unsafe_allow_html=True)
    else:
        st.write("Você posssui todas as soft skills da nossa base de dados!!")

df = pd.read_csv(db_path+"LinkedInPtFilter_atualizado.csv", sep = ";")
jobs = df['soft_skills'].apply(eval)

clusters = import_clusters()
similarity_matrix = import_similarity_matrix()
percentuais_clusters = percent_clusters(df, clusters)

map_translate = {'Adaptable': 'Adaptabilidade', 'Analytical': 'Analítico', 'Assertive': 'Assertivo',  'Collaboration': 'Colaboração', 
    'Communication (generic)': 'Comunicação (genérica)', 'Communication (oral)': 'Comunicação (oral)', 'Communication (written)': 'Comunicação (escrita)',
    'Cooperation': 'Cooperação', 'Creativity': 'Criatividade', 'Critical thinking': 'Pensamento crítico', 'Curiosity': 'Curiosidade',
    'Decision making': 'Tomada de decisão', 'Diversity': 'Diversidade', 'Dynamism': 'Dinamismo', 'Empathy': 'Empatia', 'Enthusiasm': 'Entusiasmo',
    'Flexibility': 'Flexibilidade', 'Innovation': 'Inovação', 'Interpersonal': 'Interpessoal', 'Investigative': 'Investigativo',
    'Leadership': 'Liderança', 'Mentoring': 'Mentoria', 'Negotiation': 'Negociação', 'Not mention': 'Não mencionado', 'Organization': 'Organização',
    'Planning': 'Planejamento', 'Proactive': 'Proativo', 'Problem solving': 'Resolução de problemas', 'Resilience': 'Resiliência',
    'Self disciplined': 'Auto-disciplina', 'Self management': 'Autogestão', 'Self motivated': 'Auto-motivado', 'Team': 'Trabalho em equipe'
}

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
st.set_page_config(page_title="TalentJobRadar", page_icon="./imgs/Logo.png", layout="centered")

# Carregar imagens
logo_base64 = get_image_as_base64("./imgs/Logo-No-White.png")
slogan_base64 = get_image_as_base64("./imgs/Slogan-No-White.png")


st.markdown(
    """
    <style>
    .centered-image {
        display: flex;
        justify-content: center;
    }
    .footer {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-top: 50px;
    }
    .footer img {
        width: 150px;
        margin-right: 10px;
    }
    .footer p {
        margin: 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown('<div class="centered-image" style="margin-bottom: 50px;"><img src="data:image/png;base64,{}" style="width: 300px;"></div>'.format(slogan_base64), unsafe_allow_html=True)

# st.title("TalentoJobRadar")
# Texto informativo
st.markdown("""Recomendação de empregos baseados nas suas habilidades, ou desenvolvimento de habilidades para alavancar suas possibilidades!""")

options = [map_translate.get(skill, skill) for skill in uniques if skill != 'Not mention']
options = sorted(options)

habilidades_selecionadas = st.multiselect('Selecione suas habilidades:', options)

# Slider
faixa_salarial = st.slider('Pretensão Salarial:', 1000, 10000, 5000, step=100)

# Botão
col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

botao_empregos_pressed = False
botao_habilidades_pressed = False

# Adicionar um botão em cada coluna do meio
with col2:
    if st.button('Buscar Empregos'):
        result, similaridades = RecomenderJob(df, habilidades_selecionadas, similarity_matrix, map_positions, uniques)
        link_address = "https://www.linkedin.com/jobs/view/"
        lista = []
        i = 0
        for index, row in result.iterrows():
            lista.append([row['title'], row['org_name'], row['soft_skills'], f"{link_address}{row['ID']}", similaridades[i]])
            i+=1
        botao_empregos_pressed = True

with col4:
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
    i = 0
    for item in lista:
        create_card(i, item[0], item[1], item[2], item[3], item[4], habilidades_selecionadas, map_positions, uniques)
        i+=1
if botao_habilidades_pressed:
    botao_empregos_pressed = False
    st.markdown("---")
    st.title("Resultados:")
    create_podium(habilidades_show, rank_skills, map_translate)

footer = f'''
<div class="footer">
    <img src="data:image/png;base64,{logo_base64}" alt="Logo">
    <p>© 2024 TalentJobRadar</p>
</div>
'''

st.markdown(footer, unsafe_allow_html=True)