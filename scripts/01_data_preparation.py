import pandas as pd
import numpy as np
from datetime import datetime
import os

print("=" * 60)
print("DATA PREPARATION - MOVIELENS DATASET")
print("=" * 60)

# Load ratings dataset
print("\n1. Loading original dataset...")
ratings = pd.read_csv('../data/rating.csv')
movies = pd.read_csv('../data/movie.csv')

if pd.api.types.is_numeric_dtype(ratings['timestamp']):
    ratings['date'] = pd.to_datetime(ratings['timestamp'], unit='s')
else:
    ratings['date'] = pd.to_datetime(ratings['timestamp'])

print("\n2. Filtering data for 2010-2015 period...")
ratings = ratings[(ratings['date'].dt.year >= 2010) & (ratings['date'].dt.year <= 2015)]
print(f"After time filtering:")
print(f"Remaining ratings: {len(ratings):,}")
print(f"Date range: {ratings['date'].min()} to {ratings['date'].max()}")

print(f"Original dataset (2010-2015):")
print(f"  Total ratings: {len(ratings):,}")
print(f"  Unique users: {ratings['userId'].nunique():,}")
print(f"  Unique movies: {ratings['movieId'].nunique():,}")

# Filter users with at least 50 ratings
print("\n3. Filtering users (minimum 50 ratings)...")
user_counts = ratings['userId'].value_counts()
valid_users = user_counts[user_counts >= 50].index
ratings_filtered = ratings[ratings['userId'].isin(valid_users)]

print(f"After user filtering:")
print(f"  Remaining ratings: {len(ratings_filtered):,}")
print(f"  Remaining users: {ratings_filtered['userId'].nunique():,}")
print(f"  Reduction: {(1 - len(ratings_filtered)/len(ratings)) * 100:.2f}%")

# Filter movies with at least 20 ratings
print("\n4. Filtering movies (minimum 20 ratings)...")
movie_counts = ratings_filtered['movieId'].value_counts()
valid_movies = movie_counts[movie_counts >= 20].index
ratings_filtered = ratings_filtered[ratings_filtered['movieId'].isin(valid_movies)]

print(f"After movie filtering:")
print(f"  Remaining ratings: {len(ratings_filtered):,}")
print(f"  Remaining movies: {ratings_filtered['movieId'].nunique():,}")
print(f"  Reduction from original: {(1 - len(ratings_filtered)/len(ratings)) * 100:.2f}%")

# Calculate new sparsity
n_users = ratings_filtered['userId'].nunique()
n_movies = ratings_filtered['movieId'].nunique()
n_ratings = len(ratings_filtered)
sparsity = 1 - (n_ratings / (n_users * n_movies))

print(f"\n5. New matrix statistics:")
print(f"  Matrix dimensions: {n_users:,} users × {n_movies:,} movies")
print(f"  Filled cells: {n_ratings:,}")
print(f"  Sparsity: {sparsity * 100:.4f}%")
print(f"  Density: {(1 - sparsity) * 100:.4f}%")

# Remap user and movie IDs to be contiguous (0, 1, 2, ...)
print("\n6. Remapping IDs to contiguous range...")
user_id_map = {old_id: new_id for new_id, old_id in enumerate(ratings_filtered['userId'].unique())}
movie_id_map = {old_id: new_id for new_id, old_id in enumerate(ratings_filtered['movieId'].unique())}

ratings_filtered['userId'] = ratings_filtered['userId'].map(user_id_map)
ratings_filtered['movieId'] = ratings_filtered['movieId'].map(movie_id_map)

print(f"  User IDs remapped: 0 to {ratings_filtered['userId'].max()}")
print(f"  Movie IDs remapped: 0 to {ratings_filtered['movieId'].max()}")

# Split into train and test sets (80/20 split)
print("\n7. Creating train/test split (80/20)...")
np.random.seed(42)  # For reproducibility

# Shuffle the data
ratings_shuffled = ratings_filtered.sample(frac=1, random_state=42).reset_index(drop=True)

# Split
train_size = int(0.8 * len(ratings_shuffled))
train_data = ratings_shuffled[:train_size]
test_data = ratings_shuffled[train_size:]

print(f"  Training set: {len(train_data):,} ratings ({len(train_data)/len(ratings_filtered)*100:.1f}%)")
print(f"  Test set: {len(test_data):,} ratings ({len(test_data)/len(ratings_filtered)*100:.1f}%)")

# Verify all users and movies in test set exist in train set
train_users = set(train_data['userId'].unique())
train_movies = set(train_data['movieId'].unique())
test_users = set(test_data['userId'].unique())
test_movies = set(test_data['movieId'].unique())

users_only_in_test = test_users - train_users
movies_only_in_test = test_movies - train_movies

print(f"\n8. Checking data consistency...")
print(f"  Users only in test set: {len(users_only_in_test)}")
print(f"  Movies only in test set: {len(movies_only_in_test)}")

if len(users_only_in_test) > 0 or len(movies_only_in_test) > 0:
    print("  ⚠ Warning: Some users/movies only appear in test set")
    print("  → Removing these ratings from test set...")
    test_data = test_data[
        test_data['userId'].isin(train_users) & 
        test_data['movieId'].isin(train_movies)
    ]
    print(f"  New test set size: {len(test_data):,} ratings")

# Save processed datasets
print("\n9. Saving processed datasets...")
output_dir = '../data/processed/'
os.makedirs(output_dir, exist_ok=True)

# Save full filtered dataset
ratings_filtered.to_csv(output_dir + 'ratings_filtered.csv', index=False)
print(f"  ✓ Saved: ratings_filtered.csv")

# Save train/test splits
train_data.to_csv(output_dir + 'train.csv', index=False)
test_data.to_csv(output_dir + 'test.csv', index=False)
print(f"  ✓ Saved: train.csv")
print(f"  ✓ Saved: test.csv")

# Save ID mappings for reference
user_mapping_df = pd.DataFrame(list(user_id_map.items()), columns=['original_userId', 'new_userId'])
movie_mapping_df = pd.DataFrame(list(movie_id_map.items()), columns=['original_movieId', 'new_movieId'])

user_mapping_df.to_csv(output_dir + 'user_id_mapping.csv', index=False)
movie_mapping_df.to_csv(output_dir + 'movie_id_mapping.csv', index=False)
print(f"  ✓ Saved: user_id_mapping.csv")
print(f"  ✓ Saved: movie_id_mapping.csv")

# Create a subset of movies data for reference
movies_filtered = movies[movies['movieId'].isin(movie_id_map.keys())].copy()
movies_filtered['movieId'] = movies_filtered['movieId'].map(movie_id_map)
movies_filtered.to_csv(output_dir + 'movies_filtered.csv', index=False)
print(f"  ✓ Saved: movies_filtered.csv")

# Save statistics summary
print("\n10. Saving statistics summary...")
stats = {
    'date_range_start': '2010-01-01',
    'date_range_end': '2015-12-31',
    'original_ratings': len(ratings),
    'original_users': ratings['userId'].nunique(),
    'original_movies': ratings['movieId'].nunique(),
    'filtered_ratings': len(ratings_filtered),
    'filtered_users': n_users,
    'filtered_movies': n_movies,
    'sparsity': sparsity,
    'density': 1 - sparsity,
    'train_size': len(train_data),
    'test_size': len(test_data),
    'min_user_ratings': 50,
    'min_movie_ratings': 20
}

stats_df = pd.DataFrame([stats])
stats_df.to_csv(output_dir + 'dataset_statistics.csv', index=False)
print(f"  ✓ Saved: dataset_statistics.csv")

print("\n" + "=" * 60)
print("DATA PREPARATION COMPLETED!")
print("=" * 60)
print(f"\nProcessed files saved in: {output_dir}")
