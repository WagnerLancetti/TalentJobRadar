from src.recommendation import map_vector, increase_similarity
import ast
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import base64

def get_image_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

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
    
def create_card(card_id, job_title, company, skills, link, similarity, seniority, tipo_trabalho, vector, map_positions, uniques, map_translate):
    map_seniority = {'Entry level': 'Nível de Entrada', 'Mid-level': 'Nídel Intermediário', 'Senior': 'Nível Sênior', 'Not mentioned': 'Não menciona'}
    skills_list = ast.literal_eval(skills)
    
    binary_vector = map_vector(vector, map_positions, len(uniques))
    binary_vector2 = map_vector(skills_list, map_positions, len(uniques))

    translated_skills_list = sorted([map_translate.get(skill, skill) for skill in skills_list])
    skills_formatted = ', '.join(translated_skills_list)

    image_path = "./imgs/upgrade-nobg.png"
    img_base64 = get_image_as_base64(image_path)
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
            <p><strong>Tipo de trabalho:</strong> {tipo_trabalho}</p>
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

def create_podium(habilidades_show, rank_skills, valores_percentuais, map_translate, figs):
    st.markdown("""
        <h5 style='margin-top: 30px; text-align: center;'>Recomendações de habilidades que podem melhorar suas possibilidades de empregabilidade, dados os filtros selecionados:</h5>
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
        for i, (habilidade, fig) in enumerate(zip(habilidades_show, figs), 1):
            habilidade_traduzida = map_translate.get(habilidade, habilidade)  # Obtém a tradução ou usa o original se não houver tradução
            col1, col2 = st.columns([1, 1])  # Define a proporção das colunas

            with col1:
                if i == 1:
                    st.markdown(f"""
                        <span class='first' style='display: block; text-align: center;'>
                        {i}º - A habilidade - {habilidade_traduzida} - está presente em {valores_percentuais[0]}% dos anúncios, mostrando ser de grande importância para melhorar sua empregabilidade!
                        </span>""", unsafe_allow_html=True)
                elif i == 2:
                    st.markdown(f"""
                        <span class='second' style='display: block; text-align: center;'>
                        {i}º - A habilidade - {habilidade_traduzida} - está presente em {valores_percentuais[1]}% dos anúncios. Seria interessante trabalhá-la para conseguir mais oportunidades!
                        </span>""", unsafe_allow_html=True)
                elif i == 3:
                    st.markdown(f"""
                        <span class='third' style='display: block; text-align: center;'>
                        {i}º - A última recomendação, mas não menos importante, visto que está em {valores_percentuais[2]}% dos anúncios, é a habilidade - {habilidade_traduzida} - e, desenvolvê-la, também pode ser de grande valia para conseguir um emprego!
                        </span>""", unsafe_allow_html=True)

            with col2:
                st.pyplot(fig, use_container_width=True)
    elif len(rank_skills) > 0:
        for i, ((habilidade, valor), fig) in enumerate(zip(rank_skills.items(), figs), 1):
            habilidade_traduzida = map_translate.get(habilidade, habilidade)  # Obtém a tradução ou usa o original se não houver tradução
            col1, col2 = st.columns([3, 2])  # Define a proporção das colunas

            with col1:
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

            with col2:
                st.pyplot(fig, use_container_width=True)
    else:
        st.write("Você possui todas as soft skills da nossa base de dados!!")
