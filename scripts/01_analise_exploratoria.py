import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configurações de visualização
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 60)
print("ANÁLISE EXPLORATÓRIA - MOVIELENS DATASET")
print("=" * 60)

# Carregar os datasets
print("\n1. Carregando datasets...")
try:
    ratings = pd.read_csv('../dados/rating.csv')
    movies = pd.read_csv('../dados/movie.csv')
    tags = pd.read_csv('../dados/tag.csv')
    links = pd.read_csv('../dados/link.csv')
    genome_scores = pd.read_csv('../dados/genome_scores.csv')
    genome_tags = pd.read_csv('../dados/genome_tags.csv')
    print("✓ Todos os arquivos carregados com sucesso!")
except Exception as e:
    print(f"✗ Erro ao carregar arquivos: {e}")
    exit(1)

# Informações básicas
print("\n2. INFORMAÇÕES BÁSICAS DO DATASET")
print("-" * 60)
print(f"Total de avaliações: {len(ratings):,}")
print(f"Total de filmes: {len(movies):,}")
print(f"Total de usuários únicos: {ratings['userId'].nunique():,}")
print(f"Total de tags aplicadas: {len(tags):,}")
print(f"Total de genome tags: {len(genome_tags):,}")
print(f"Total de genome scores: {len(genome_scores):,}")

# Estatísticas das avaliações
print("\n3. ESTATÍSTICAS DAS AVALIAÇÕES")
print("-" * 60)
print(ratings['rating'].describe())
print(f"\nDistribuição das notas:")
print(ratings['rating'].value_counts().sort_index())

# Converter timestamp para data
ratings['date'] = pd.to_datetime(ratings['timestamp'], unit='s')
print(f"\nPeríodo das avaliações:")
print(f"  Primeira avaliação: {ratings['date'].min()}")
print(f"  Última avaliação: {ratings['date'].max()}")
print(f"  Período total: {(ratings['date'].max() - ratings['date'].min()).days} dias")

# Análise de usuários
print("\n4. ANÁLISE DE USUÁRIOS")
print("-" * 60)
user_stats = ratings.groupby('userId').agg({
    'rating': ['count', 'mean']
}).round(2)
user_stats.columns = ['num_ratings', 'avg_rating']
print(f"Média de avaliações por usuário: {user_stats['num_ratings'].mean():.2f}")
print(f"Mediana de avaliações por usuário: {user_stats['num_ratings'].median():.2f}")
print(f"Usuário mais ativo: {user_stats['num_ratings'].max()} avaliações")
print(f"Usuário menos ativo: {user_stats['num_ratings'].min()} avaliações")

# Análise de filmes
print("\n5. ANÁLISE DE FILMES")
print("-" * 60)
movie_stats = ratings.groupby('movieId').agg({
    'rating': ['count', 'mean']
}).round(2)
movie_stats.columns = ['num_ratings', 'avg_rating']
print(f"Média de avaliações por filme: {movie_stats['num_ratings'].mean():.2f}")
print(f"Mediana de avaliações por filme: {movie_stats['num_ratings'].median():.2f}")
print(f"Filme mais avaliado: {movie_stats['num_ratings'].max()} avaliações")
print(f"Filmes com apenas 1 avaliação: {(movie_stats['num_ratings'] == 1).sum()}")

# Sparsity (dispersão) da matriz
print("\n6. SPARSITY DA MATRIZ")
print("-" * 60)
n_users = ratings['userId'].nunique()
n_movies = ratings['movieId'].nunique()
n_ratings = len(ratings)
sparsity = 1 - (n_ratings / (n_users * n_movies))
print(f"Dimensões da matriz: {n_users:,} usuários × {n_movies:,} filmes")
print(f"Células possíveis: {n_users * n_movies:,}")
print(f"Células preenchidas: {n_ratings:,}")
print(f"Sparsity: {sparsity * 100:.4f}%")
print(f"Densidade: {(1 - sparsity) * 100:.4f}%")

# Top 10 filmes mais avaliados
print("\n7. TOP 10 FILMES MAIS AVALIADOS")
print("-" * 60)
top_movies = movie_stats.nlargest(10, 'num_ratings')
top_movies_info = top_movies.merge(movies, on='movieId')
for idx, row in top_movies_info.iterrows():
    print(f"{row['title']}: {row['num_ratings']:.0f} avaliações (média: {row['avg_rating']:.2f})")

# Criar visualizações
print("\n8. Gerando visualizações...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Gráfico 1: Distribuição de notas
axes[0, 0].hist(ratings['rating'], bins=10, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Distribuição das Notas', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Nota')
axes[0, 0].set_ylabel('Frequência')
axes[0, 0].grid(alpha=0.3)

# Gráfico 2: Avaliações por usuário (log scale)
user_counts = ratings['userId'].value_counts()
axes[0, 1].hist(user_counts, bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Distribuição de Avaliações por Usuário', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Número de Avaliações')
axes[0, 1].set_ylabel('Número de Usuários')
axes[0, 1].set_yscale('log')
axes[0, 1].grid(alpha=0.3)

# Gráfico 3: Avaliações por filme (log scale)
movie_counts = ratings['movieId'].value_counts()
axes[1, 0].hist(movie_counts, bins=50, edgecolor='black', alpha=0.7, color='orange')
axes[1, 0].set_title('Distribuição de Avaliações por Filme', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Número de Avaliações')
axes[1, 0].set_ylabel('Número de Filmes')
axes[1, 0].set_yscale('log')
axes[1, 0].grid(alpha=0.3)

# Gráfico 4: Avaliações ao longo do tempo
ratings_per_year = ratings.groupby(ratings['date'].dt.year).size()
axes[1, 1].plot(ratings_per_year.index, ratings_per_year.values, marker='o', linewidth=2)
axes[1, 1].set_title('Avaliações ao Longo do Tempo', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Ano')
axes[1, 1].set_ylabel('Número de Avaliações')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../graficos/analise_exploratoria.png', dpi=300, bbox_inches='tight')
print("✓ Gráficos salvos em: ../graficos/analise_exploratoria.png")

print("\n" + "=" * 60)
print("ANÁLISE CONCLUÍDA!")
print("=" * 60)
print("\nPróximos passos:")
print("1. Analisar os gráficos gerados")
print("2. Preparar subconjunto dos dados para experimentos")
print("3. Implementar os modelos de recomendação")
