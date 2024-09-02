import pandas as pd
import src.clustering
import numpy as np
import matplotlib.pyplot as plt

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

def RelacaoSenioridade(df, vector, recomendacoes, map_translate, selected_levels, selected_types):
    cores = ['gold', 'gray', '#cd7f32']  # Definir cores para pódio
    figs = []
    for i, (item, valor) in enumerate(recomendacoes.items()):
        if i > 2:
            break
        df_filtrado = df[df['soft_skills'].apply(lambda x: item in x)]

        # Remover linhas com 'Not mentioned'
        df_filtrado = df_filtrado[df_filtrado['seniority'] != 'Not mentioned']

        df_filtrado_ordenado = df_filtrado.sort_values(by='seniority', ascending=False)

        if not df_filtrado_ordenado.empty:
            # Adiciona as senioridades e contagens para o gráfico de radar
            senioridade_contagem = df_filtrado_ordenado['seniority'].value_counts().sort_index()
            figs.append(plot_radar_chart(senioridade_contagem, item, cores[i], map_translate))
    return figs

def plot_radar_chart(senioridade_contagem, skill_name, cor, map_translate):
    map_seniority = {'Entry level': 'Nível de Entrada', 'Mid-level': 'Nídel Intermediário', 'Senior': 'Nível Sênior', 'Not mentioned': 'Não menciona'}
    categories = [map_seniority.get(cat, cat) for cat in senioridade_contagem.index.tolist()]
    N = len(categories)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    values = senioridade_contagem.tolist()
    values += values[:1]
    skill_name = map_translate.get(skill_name, skill_name)

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=1, linestyle='solid',color=cor, label=skill_name)
    ax.fill(angles, values, alpha=0.25, color=cor)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, horizontalalignment='center',verticalalignment='top')
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.05))  # Ajustar a posição da legenda
    ax.set_title(f"Distribuição de Empregos dada a especificação de Senioridade para {skill_name}", pad=30)  # Aumentar a margem do título
    # Adicionar os valores ao redor do gráfico
    for i, (angle, value) in enumerate(zip(angles, values)):
        if i < len(categories):  # Evitar adicionar o valor repetido do fechamento do loop
            x = np.cos(angle) * (1.1)  # Ajustar a posição do número
            y = np.sin(angle) * (1.1)
            ax.text(angle, value, str(value), horizontalalignment='center', verticalalignment='bottom', size=10, color='#000080', weight='semibold')

    return fig

def RecomenderJob(df, vector, similarity_matrix, map_positions, uniques, selected_levels, selected_types):
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
        
        # Nível de senioridade e o tipo de trabalho estão nos filtros selecionados
        seniority_match = (df.iloc[idx]['seniority'] in selected_levels) or not selected_levels
        remote_match = (df.iloc[idx]['remote'] in selected_types) or not selected_types
        
        # Ambos os filtros aceitos
        if seniority_match and remote_match:
            indices.append(idx)

    
    selected_rows = df.loc[indices]
    similarity_rank = similarities[indices]

    return selected_rows, similarity_rank


def RecomenderSkill(df, vector, uniques, map_positions, binary_vectors, percentuais_clusters, map_translate, selected_levels, selected_types):
    binary_vector = map_vector(vector, map_positions, len(uniques))
    similaridades = []
    # Binary_vectors = vetores binários dos clusters
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

            figs = RelacaoSenioridade(df, vector, habilidades_ordenadas, map_translate, selected_levels, selected_types)

            return habilidades_ordenadas, rank_filtrado, valores_percentuais, figs

    # Se nenhum cluster tiver pelo menos 3 habilidades recomendadas, retornar vazio
    return {}, {}, [], []