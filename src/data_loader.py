import pandas as pd
from pathlib import Path
from src.clustering import import_clusters, import_similarity_matrix, create_binary_vector 
from src.filtering import percent_clusters


def load_data(db_path):
    db_file = Path(db_path) / "LinkedInPt.csv"
    df = pd.read_csv(db_file, sep=";")
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

    uniques = sorted(set(skill for skills in clusters.values() for skill in skills) | {'Not mention'})
    
    map_positions = {item: idx for idx, item in enumerate(uniques)}
    vector_size = len(uniques)
    binary_vectors = {cluster: create_binary_vector(skills, map_positions, vector_size) for cluster, skills in clusters.items()}
    return df, clusters, similarity_matrix, percentuais_clusters, uniques, map_positions, binary_vectors