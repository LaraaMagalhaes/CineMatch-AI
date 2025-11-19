import pandas as pd
import os
# Definindo o caminho do arquivo
csv_path = os.path.join("data", "tmdb_5000_movies.csv")
try:
    # Lendo o CSV
    df = pd.read_csv(csv_path)
    
    print("O Dataset foi carregado.")
    print(f"Temos {df.shape[0]} filmes e {df.shape[1]} colunas.")
    
    # Mostra os primeiros 2 filmes só pra ver a cara dos dados
    print("\nExemplo dos dados:")
    print(df[['title', 'overview']].head(2))

except FileNotFoundError:
    print(" Erro")