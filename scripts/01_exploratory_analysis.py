import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Visualization settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 60)
print("EXPLORATORY ANALYSIS - MOVIELENS DATASET")
print("=" * 60)

# Load datasets
print("\n1. Loading datasets...")
try:
    ratings = pd.read_csv('../dados/rating.csv')
    movies = pd.read_csv('../dados/movie.csv')
    tags = pd.read_csv('../dados/tag.csv')
    links = pd.read_csv('../dados/link.csv')
    genome_scores = pd.read_csv('../dados/genome_scores.csv')
    genome_tags = pd.read_csv('../dados/genome_tags.csv')
    print("✓ All files loaded successfully!")
except Exception as e:
    print(f"✗ Error loading files: {e}")
    exit(1)

# Basic information
print("\n2. BASIC DATASET INFORMATION")
print("-" * 60)
print(f"Total ratings: {len(ratings):,}")
print(f"Total movies: {len(movies):,}")
print(f"Total unique users: {ratings['userId'].nunique():,}")
print(f"Total tags applied: {len(tags):,}")
print(f"Total genome tags: {len(genome_tags):,}")
print(f"Total genome scores: {len(genome_scores):,}")

# Rating statistics

print("\n3. RATING STATISTICS")
print("-" * 60)
print(ratings['rating'].describe())
print(f"\nRating distribution:")
print(ratings['rating'].value_counts().sort_index())

# Convert timestamp to date
# Check if timestamp is numeric or string
if pd.api.types.is_numeric_dtype(ratings['timestamp']):
    ratings['date'] = pd.to_datetime(ratings['timestamp'], unit='s')
else:
    ratings['date'] = pd.to_datetime(ratings['timestamp'])

print(f"\nRating period:")
print(f"  First rating: {ratings['date'].min()}")
print(f"  Last rating: {ratings['date'].max()}")
print(f"  Total period: {(ratings['date'].max() - ratings['date'].min()).days} days")

# User analysis

print("\n4. USER ANALYSIS")
print("-" * 60)
user_stats = ratings.groupby('userId').agg({
    'rating': ['count', 'mean']
}).round(2)
user_stats.columns = ['num_ratings', 'avg_rating']
print(f"Average ratings per user: {user_stats['num_ratings'].mean():.2f}")
print(f"Median ratings per user: {user_stats['num_ratings'].median():.2f}")
print(f"Most active user: {user_stats['num_ratings'].max()} ratings")
print(f"Least active user: {user_stats['num_ratings'].min()} ratings")

# Movie analysis
print("\n5. MOVIE ANALYSIS")
print("-" * 60)
movie_stats = ratings.groupby('movieId').agg({
    'rating': ['count', 'mean']
}).round(2)
movie_stats.columns = ['num_ratings', 'avg_rating']
print(f"Average ratings per movie: {movie_stats['num_ratings'].mean():.2f}")
print(f"Median ratings per movie: {movie_stats['num_ratings'].median():.2f}")
print(f"Most rated movie: {movie_stats['num_ratings'].max()} ratings")
print(f"Movies with only 1 rating: {(movie_stats['num_ratings'] == 1).sum()}")

# Matrix sparsity
print("\n6. MATRIX SPARSITY")
print("-" * 60)
n_users = ratings['userId'].nunique()
n_movies = ratings['movieId'].nunique()
n_ratings = len(ratings)
sparsity = 1 - (n_ratings / (n_users * n_movies))
print(f"Matrix dimensions: {n_users:,} users × {n_movies:,} movies")
print(f"Possible cells: {n_users * n_movies:,}")
print(f"Filled cells: {n_ratings:,}")
print(f"Sparsity: {sparsity * 100:.4f}%")
print(f"Density: {(1 - sparsity) * 100:.4f}%")

# Top 10 most rated movies
print("\n7. TOP 10 MOST RATED MOVIES")
print("-" * 60)
top_movies = movie_stats.nlargest(10, 'num_ratings')
top_movies_info = top_movies.merge(movies, on='movieId')
for idx, row in top_movies_info.iterrows():
    print(f"{row['title']}: {row['num_ratings']:.0f} ratings (avg: {row['avg_rating']:.2f})")

# Create visualizations
print("\n8. Generating visualizations...")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Chart 1: Rating distribution
axes[0, 0].hist(ratings['rating'], bins=10, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('Rating Distribution', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Rating')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].grid(alpha=0.3)

# Chart 2: Ratings per user (log scale)
user_counts = ratings['userId'].value_counts()
axes[0, 1].hist(user_counts, bins=50, edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Ratings per User Distribution', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Number of Ratings')
axes[0, 1].set_ylabel('Number of Users')
axes[0, 1].set_yscale('log')
axes[0, 1].grid(alpha=0.3)

# Chart 3: Ratings per movie (log scale)
movie_counts = ratings['movieId'].value_counts()
axes[1, 0].hist(movie_counts, bins=50, edgecolor='black', alpha=0.7, color='orange')
axes[1, 0].set_title('Ratings per Movie Distribution', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Number of Ratings')
axes[1, 0].set_ylabel('Number of Movies')
axes[1, 0].set_yscale('log')
axes[1, 0].grid(alpha=0.3)

# Chart 4: Ratings over time
ratings_per_year = ratings.groupby(ratings['date'].dt.year).size()
axes[1, 1].plot(ratings_per_year.index, ratings_per_year.values, marker='o', linewidth=2)
axes[1, 1].set_title('Ratings Over Time', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Year')
axes[1, 1].set_ylabel('Number of Ratings')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()

plt.savefig('../graficos/exploratory_analysis.png', dpi=300, bbox_inches='tight')

print("✓ Charts saved to: ../graficos/exploratory_analysis.png")
print("\n" + "=" * 60)
print("ANALYSIS COMPLETED!")
print("=" * 60)
print("\nNext steps:")
print("1. Analyze generated charts")
print("2. Prepare data subset for experiments")
print("3. Implement recommendation models")
