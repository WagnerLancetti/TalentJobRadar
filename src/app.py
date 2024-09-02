from src.data_loader import load_data
from src.interface import get_image_as_base64, create_card, create_podium
from src.filtering import create_uniques, create_clusters
from src.recommendation import RecomenderJob, RecomenderSkill
import streamlit as st

def run():
    df, clusters, similarity_matrix, percentuais_clusters, uniques, map_positions, binary_vector = load_data("./db/")
    
    map_translate = {'Adaptable': 'Adaptabilidade', 'Analytical': 'Analítico', 'Assertive': 'Assertivo',  'Collaboration': 'Colaboração', 
    'Communication (generic)': 'Comunicação (genérica)', 'Communication (oral)': 'Comunicação (oral)', 'Communication (written)': 'Comunicação (escrita)',
    'Cooperation': 'Cooperação', 'Creativity': 'Criatividade', 'Critical thinking': 'Pensamento crítico', 'Curiosity': 'Curiosidade',
    'Decision making': 'Tomada de decisão', 'Diversity': 'Diversidade', 'Dynamism': 'Dinamismo', 'Empathy': 'Empatia', 'Enthusiasm': 'Entusiasmo',
    'Flexibility': 'Flexibilidade', 'Innovation': 'Inovação', 'Interpersonal': 'Interpessoal', 'Investigative': 'Investigativo',
    'Leadership': 'Liderança', 'Mentoring': 'Mentoria', 'Negotiation': 'Negociação', 'Not mention': 'Não mencionado', 'Organization': 'Organização',
    'Planning': 'Planejamento', 'Proactive': 'Proativo', 'Problem solving': 'Resolução de problemas', 'Resilience': 'Resiliência',
    'Self disciplined': 'Auto-disciplina', 'Self management': 'Autogestão', 'Self motivated': 'Auto-motivado', 'Team': 'Trabalho em equipe'
    }

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
    translation_seniority = {
        'Entry level': 'Entrada',
        'Mid-level': 'Intermediário',
        'Senior': 'Sênior'
    }

    reverse_translation_seniority = {v: k for k, v in translation_seniority.items()}

    st.sidebar.write("Aplicar para as seguintes senioridades: ")
    filtered_uniques = sorted([level for level in df['seniority'].unique() if level != 'Not mentioned'])

    selected_levels_translated = []
    for level in filtered_uniques:
        translated_level = translation_seniority.get(level, level)
        if st.sidebar.checkbox(translated_level):
            selected_levels_translated.append(translated_level)

    # Converter os níveis selecionados de volta para inglês
    selected_levels = [reverse_translation_seniority.get(level, level) for level in selected_levels_translated]

    # Remoto ou não
    type_translation_dict = {'Remoto': 'Yes',
        'Presencial': 'No'}
    type_job = ['Remoto', 'Presencial']
    st.sidebar.write("Formato de trabalho: ")
    opcoes = {opcao: st.sidebar.checkbox(opcao, value=False) for opcao in type_job}

    # Botão
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

    botao_empregos_pressed = False
    botao_habilidades_pressed = False

    # Adicionar um botão em cada coluna do meio
    with col2:
        if st.button('Buscar Empregos'):
            selected_types = [type_translation_dict[opcao] for opcao, selecionado in opcoes.items() if selecionado]
            habilidades_selecionadas = [skill for skill in uniques if map_translate.get(skill, skill) in habilidades_selecionadas] # voltar para ingles
            result, similaridades = RecomenderJob(df, habilidades_selecionadas, similarity_matrix, map_positions, uniques, selected_levels, selected_types)
            link_address = "https://www.linkedin.com/jobs/view/"
            lista = []
            i = 0
            for index, row in result.iterrows():
                tipo_trabalho = 'Remoto' if row['remote'] == 'Yes' else 'Presencial'
                lista.append([row['title'], row['org_name'], row['soft_skills'], f"{link_address}{row['ID']}", similaridades[i], row['seniority'], tipo_trabalho])
                i+=1
            botao_empregos_pressed = True

    with col4:
        if st.button('Melhorar Habilidades'):
            habilidades_selecionadas = [skill for skill in uniques if map_translate.get(skill, skill) in habilidades_selecionadas]
            selected_types = [type_translation_dict[opcao] for opcao, selecionado in opcoes.items() if selecionado]
            clusters_filter, percentuais_filter = create_clusters(selected_levels,selected_types)
            uniques_filter, map_positions_filter, binary_vectors_filter = create_uniques(clusters)
            
            habilidades_result, rank_skills, valores_percentuais, figs = RecomenderSkill(df, habilidades_selecionadas, uniques_filter, map_positions_filter, binary_vectors_filter, percentuais_filter, map_translate, selected_levels, selected_types)
            if habilidades_result != {} and rank_skills != {}:
                habilidades_show = []
                for i, habilidade in enumerate(habilidades_result, 1):
                    if i > 3:
                        break
                    habilidades_show.append(habilidade)
                botao_habilidades_pressed = True
            else:
                st.markdown("---")
                st.markdown("<h1 style='text-align: center;'>Resultados</h1>", unsafe_allow_html=True)
                st.markdown("<h4 style='text-align: center;'>😞 Não há habilidades para recomendar!😞</h4>", unsafe_allow_html=True)

    if botao_empregos_pressed:
        botao_habilidades_pressed = False
        st.markdown("---")
        if(len(lista) != 0):
            st.markdown("<h1 style='text-align: center;'>Resultados</h1>", unsafe_allow_html=True)
            cols = st.columns(2, gap="small")
            for i, item in enumerate(lista):
                col = cols[i % 2]
                with col:
                    create_card(i, item[0], item[1], item[2], item[3], item[4], item[5], item[6], habilidades_selecionadas, map_positions, uniques, map_translate)
        else:
            st.markdown("<h4 style='text-align: center;'>😞 Não foram encontrados empregos para esses filtros! 😞</h4>", unsafe_allow_html=True)
    if botao_habilidades_pressed:
        botao_empregos_pressed = False
        st.markdown("---")
        st.markdown("<h1 style='text-align: center;'>Resultados</h1>", unsafe_allow_html=True)
        create_podium(habilidades_show, rank_skills, valores_percentuais, map_translate, figs)

    footer = f'''
    <div class="footer">
        <img src="data:image/png;base64,{logo_base64}" alt="Logo">
        <p>© 2024 TalentJobRadar</p>
    </div>
    '''

    st.markdown(footer, unsafe_allow_html=True)

if __name__ == "__main__":
    run()