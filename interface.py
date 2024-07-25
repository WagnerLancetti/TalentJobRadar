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


def jaccard_similarity(A, B):
    A = np.array(A, dtype=np.int32)  # Converter A para array numérico
    B = np.array(B, dtype=np.int32)  # Converter B para array numérico

    intersection = np.sum(np.logical_and(A, B))
    union = np.sum(np.logical_or(A, B))
    
    return intersection / union if union > 0 else 0  # Evita divisão por zero

def increase_similarity(A, B, map_positions, map_translate):
    old_similarity = jaccard_similarity(A, B)
    ids = []
    for i in range(len(A)):
        if B[i] == 1 and A[i] == 0:
            A[i] = 1  # adiciona a habilidade em A
            ids.append(i)
    str_return = ""
    new_similarity = jaccard_similarity(A, B)
    if new_similarity > old_similarity:
        str_return = f"É recomendado trabalhar as seguintes habilidades para melhorar sua aptidão para {new_similarity*100:.2f}% nessa vaga: "
        added_skills = [skill for skill, idx in map_positions.items() if idx in ids]
        added_skills = [map_translate.get(skill, skill) for skill in added_skills]
        str_return += ", ".join(added_skills)
    else:
        str_return = "Você já está o melhor preparado possível para esse emprego!"
    return str_return

def ranking_skills(percentuais_clusters):
    soma_habilidades = {}

    for subdict in percentuais_clusters.values(): # percorre os clusters
        for habilidade, valor in subdict.items(): # pega os valores
            soma_habilidades[habilidade] = valor

    ranking = sorted(soma_habilidades.items(), key=lambda x: x[1], reverse=True)
    return ranking

def RecomenderJob(df, vector, similarity_matrix, map_positions, uniques, selected_levels):
    binary_vector = map_vector(vector, map_positions, len(uniques))  # Vetor que será recebido da aplicação

    similarities = []
    for row in similarity_matrix:
        similarity = jaccard_similarity(binary_vector, row)
        similarities.append(similarity)

    similarities = np.array(similarities)
    index = np.argsort(similarities)[::-1]  # Índices ordenados por similaridade decrescente

    sorted_indices = index[~np.isnan(similarities[index])]
    indices = []
    for i, idx in enumerate(sorted_indices):
        if i >= 10:
            break
        if df.iloc[idx]['seniority'] in selected_levels:
            indices.append(idx)
        if selected_levels == []: # Caso nada seja selecionado
            indices.append(idx)
    
    selected_rows = df.loc[indices]
    similarity_rank = similarities[indices]

    return selected_rows, similarity_rank


def RecomenderSkill(vector, uniques, map_positions, binary_vectors, percentuais_clusters):
    binary_vector = map_vector(vector, map_positions, len(uniques))
    similaridades = []

    for elem in binary_vectors.values():
        similaridades.append(jaccard_similarity(elem, binary_vector))

    ranking_indices = np.argsort(similaridades)[::-1]

    cluster_ranking = [list(binary_vectors.keys())[i] for i in ranking_indices]

    for cluster_id in cluster_ranking:
        resultado_cluster = percentuais_clusters[cluster_id]

        # Filtrar habilidades que o usuário já possui
        habilidades_filtradas = {k: v for k, v in resultado_cluster.items() if k not in vector}
        # Ordenar habilidades por relevância
        habilidades_ordenadas = dict(sorted(habilidades_filtradas.items(), key=lambda item: item[1], reverse=True))

        # Verificar se há pelo menos 3 habilidades recomendadas
        if len(habilidades_ordenadas) >= 3:
            rank_skills = ranking_skills(percentuais_clusters)
            rank_filtrado = {k: v for k, v in rank_skills if k not in vector}
            valores_percentuais = [v for k, v in habilidades_ordenadas.items()]

            return habilidades_ordenadas, rank_filtrado, valores_percentuais

    # Se nenhum cluster tiver pelo menos 3 habilidades recomendadas, retornar vazio
    return {}, {}, []


def get_base64_of_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def create_card(card_id, job_title, company, skills, link, similarity, seniority, vector, map_positions, uniques, map_translate):
    map_seniority = {'Entry level': 'Nível de Entrada', 'Mid-level': 'Nídel Intermediário', 'Senior': 'Nível Sênior', 'Not mentioned': 'Não menciona'}
    skills_list = ast.literal_eval(skills)
    
    binary_vector = map_vector(vector, map_positions, len(uniques))
    binary_vector2 = map_vector(skills_list, map_positions, len(uniques))

    translated_skills_list = sorted([map_translate.get(skill, skill) for skill in skills_list])
    skills_formatted = ', '.join(translated_skills_list)

    image_path = "./imgs/upgrade-nobg.png"
    img_base64 = get_base64_of_image(image_path)
    num = f"{similarity*100:.2f}%"

    result = increase_similarity(binary_vector, binary_vector2, map_positions, map_translate)

    seniority = map_seniority.get(seniority,seniority)
    # HTML do card
    if result and result != "Você já está o melhor preparado possível para esse emprego!":
        result_html = f"<p style='color: #00ff95; margin-top: 25px;'>{result}</p>"
    else:
        result_html = f"<p style='margin-top: 0px;'></p>"

    # HTML do card
    card_html = f"""
     <div id="card_{card_id}" style='border: 1px solid #ddd; border-radius: 10px; padding: 16px; margin: 16px; 
                box-shadow: 2px 2px 12px rgba(0, 0, 0, 0.1); width: calc(100% - 32px); min-height: 300px; display: flex; flex-direction: column;'>
        <div style='flex: 1;'>
            <h3>{job_title}</h3>
            <p><strong>Empresa:</strong> {company}</p>
            <p><strong>Habilidades:</strong> {skills_formatted}</p>
            <p><strong>Senioridade:</strong> {seniority}</p>
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


def create_podium(habilidades_show, rank_skills, valores_percentuais, map_translate):
    st.markdown("""
        <h5 style='margin-top: 30px; text-align: center;'>Recomendações de habilidades que podem melhorar suas possibilidades de empregabilidade:</h5>
        """, unsafe_allow_html=True)
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
                st.markdown(f"""
                    <span class='first' style='display: block; margin-bottom: 20px;'>
                    {i}º - A habilidade - {habilidade_traduzida} - está presente em {valores_percentuais[0]}% dos anúncios, mostrando ser de grande importância para melhorar sua empregabilidade!
                    </span>""", unsafe_allow_html=True)
            elif i == 2:
                st.markdown(f"""
                    <span class='second' style='display: block; margin-bottom: 20px;'>
                    {i}º - A habilidade - {habilidade_traduzida} - está presente em {valores_percentuais[1]}% dos anúncios. Seria interessante trabalhá-la para conseguir mais oportunidades!
                    </span>""", unsafe_allow_html=True)
            elif i == 3:
                st.markdown(f"""
                    <span class='third' style='display: block; margin-bottom: 20px;'>
                    {i}º - A última recomendação, mas não menos importante, visto que está em {valores_percentuais[2]}% dos anúncios, é a habilidade - {habilidade_traduzida} - e, desenvolvê-la, também pode ser de grande valia para conseguir um emprego!
                    </span>""", unsafe_allow_html=True)
    elif len(rank_skills) > 0:
        for i, (habilidade, valor) in enumerate(rank_skills.items(), 1):
            habilidade_traduzida = map_translate.get(habilidade, habilidade)  # Obtém a tradução ou usa o original se não houver tradução
            if i == 1:
                st.markdown(f"""
                    <span class='first' style='display: block; margin-bottom: 20px;'>
                    {i}º - A habilidade {habilidade_traduzida} está presente em {valores_percentuais[0]}% dos anúncios, mostrando ser de grande importância para melhorar sua empregabilidade!
                    </span>""", unsafe_allow_html=True)
            elif i == 2:
                st.markdown(f"""
                    <span class='second' style='display: block; margin-bottom: 20px;'>
                    {i}º - A habilidade {habilidade_traduzida} está presente em {valores_percentuais[1]}% dos anúncios. Seria interessante trabalhá-la para conseguir mais oportunidades!
                    </span>""", unsafe_allow_html=True)
            elif i == 3:
                st.markdown(f"""
                    <span class='third' style='display: block; margin-bottom: 20px;'>
                    {i}º - A última recomendação, mas não menos importante, visto que está em {valores_percentuais[2]}% dos anúncios, é a habilidade {habilidade_traduzida}, e também pode ser de grande valia para conseguir um emprego!
                    </span>""", unsafe_allow_html=True)
    else:
        st.write("Você posssui todas as soft skills da nossa base de dados!!")




df = pd.read_csv(db_path+"LinkedInPt.csv", sep = ";")
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
st.set_page_config(page_title="TalentJobRadar", page_icon="./imgs/Logo.png", layout="wide")

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
        margin-right: 70px;
    }
    .footer img {
        width: 150px;
    }
    .footer p {
        margin: 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown('<div class="centered-image" style="margin-bottom: 50px;"><img src="data:image/png;base64,{}" style="width: 350px;"></div>'.format(slogan_base64), unsafe_allow_html=True)

st.markdown("""
        <h5 style='margin-top: 10px; text-align: center; margin-bottom: 20px'>Recomendação de empregos baseados nas suas habilidades, ou desenvolvimento de habilidades para alavancar suas possibilidades!</h5>
        """, unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)  # Adiciona uma linha em branco



# ------- SIDE BAR ------- #
st.sidebar.markdown("<h1 style='text-align: center;'>TalentJobRadar</h1>", unsafe_allow_html=True)
options = [map_translate.get(skill, skill) for skill in uniques if skill != 'Not mention']
options = sorted(options)

habilidades_selecionadas = st.sidebar.multiselect(
    'Selecione suas habilidades:',
    options
)

# Filtro de senioridade
st.sidebar.write("Aplicar para as seguintes senioridades: ")
filtered_uniques = sorted([level for level in df['seniority'].unique() if level != 'Not mentioned'])

selected_levels = []
for level in filtered_uniques:
    if st.sidebar.checkbox(level):
        selected_levels.append(level)



# Botão
col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

botao_empregos_pressed = False
botao_habilidades_pressed = False

# Adicionar um botão em cada coluna do meio
with col2:
    if st.button('Buscar Empregos'):
        habilidades_selecionadas = [skill for skill in uniques if map_translate.get(skill, skill) in habilidades_selecionadas]
        #Filtrar Df dado o valor da variável faixa_salarial
        result, similaridades = RecomenderJob(df, habilidades_selecionadas, similarity_matrix, map_positions, uniques, selected_levels)
        link_address = "https://www.linkedin.com/jobs/view/"
        lista = []
        i = 0
        for index, row in result.iterrows():
            lista.append([row['title'], row['org_name'], row['soft_skills'], f"{link_address}{row['ID']}", similaridades[i], row['seniority']])
            i+=1
        botao_empregos_pressed = True

with col4:
    if st.button('Melhorar Habilidades'):
        habilidades_selecionadas = [skill for skill in uniques if map_translate.get(skill, skill) in habilidades_selecionadas]
        habilidades_result, rank_skills, valores_percentuais = RecomenderSkill(habilidades_selecionadas, uniques, map_positions, binary_vectors, percentuais_clusters)
        if habilidades_result != {} and rank_skills != {}:
            habilidades_show = []
            for i, habilidade in enumerate(habilidades_result, 1):
                if i > 3:
                    break
                habilidades_show.append(habilidade)
            botao_habilidades_pressed = True
        else:
            st.markdown("<h4 style='text-align: center;'>😞 Não há habilidades para recomendar! 😞</h4>", unsafe_allow_html=True)

if botao_empregos_pressed:
    botao_habilidades_pressed = False
    st.markdown("---")
    if(len(lista) != 0):
        st.markdown("<h1 style='text-align: center;'>Resultados</h1>", unsafe_allow_html=True)
        cols = st.columns(2, gap="small")
        for i, item in enumerate(lista):
            col = cols[i % 2]
            with col:
                create_card(i, item[0], item[1], item[2], item[3], item[4], item[5], habilidades_selecionadas, map_positions, uniques, map_translate)
    else:
        st.markdown("<h4 style='text-align: center;'>😞 Não foram encontrados empregos para essa senioridade! 😞</h4>", unsafe_allow_html=True)
if botao_habilidades_pressed:
    botao_empregos_pressed = False
    st.markdown("---")
    st.markdown("<h1 style='text-align: center;'>Resultados</h1>", unsafe_allow_html=True)
    create_podium(habilidades_show, rank_skills, valores_percentuais, map_translate)

footer = f'''
<div class="footer">
    <img src="data:image/png;base64,{logo_base64}" alt="Logo">
    <p>© 2024 TalentJobRadar</p>
</div>
'''

st.markdown(footer, unsafe_allow_html=True)