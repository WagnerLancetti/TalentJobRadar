import os

# Defina o caminho relativo para o arquivo
db_path = "./db/clusters.zip"

# Verifique se o arquivo existe
if os.path.exists(db_path):
    print(f"Arquivo encontrado: {db_path}")
else:
    print(f"Arquivo não encontrado: {db_path}")