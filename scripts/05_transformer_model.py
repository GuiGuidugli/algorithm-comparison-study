import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import json
import os
from tqdm import tqdm

print("=" * 60)
print("TRANSFORMER MODEL")
print("=" * 60)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# Load processed data
print("\n1. Loading processed datasets...")
train_df = pd.read_csv('../data/processed/train.csv')
test_df = pd.read_csv('../data/processed/test.csv')

print(f"Training set: {len(train_df):,} ratings")
print(f"Test set: {len(test_df):,} ratings")
print(f"Users: {train_df['userId'].nunique():,}")
print(f"Movies: {train_df['movieId'].nunique():,}")

n_users = train_df['userId'].max() + 1
n_movies = train_df['movieId'].max() + 1

print(f"\nMatrix dimensions: {n_users} users × {n_movies} movies")

# ============================================================================
# Simplified Transformer Architecture for Recommendations
# ============================================================================

class TransformerRecommender(nn.Module):
    """
    Simplified Transformer for rating prediction
    Uses embeddings + self-attention to capture user-item interactions
    """
    def __init__(self, n_users, n_movies, embedding_dim=64, n_heads=4, n_layers=2, dropout=0.1):
        super(TransformerRecommender, self).__init__()
        
        # Embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.movie_embedding = nn.Embedding(n_movies, embedding_dim)
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim * 2,  # Concatenated user + movie embeddings
            nhead=n_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output layers
        self.fc1 = nn.Linear(embedding_dim * 2, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.movie_embedding.weight, std=0.01)
    
    def forward(self, user_ids, movie_ids):
        # Get embeddings
        user_emb = self.user_embedding(user_ids)  # [batch_size, embedding_dim]
        movie_emb = self.movie_embedding(movie_ids)  # [batch_size, embedding_dim]
        
        # Concatenate embeddings
        combined = torch.cat([user_emb, movie_emb], dim=-1)  # [batch_size, embedding_dim * 2]
        
        # Add sequence dimension for transformer
        combined = combined.unsqueeze(1)  # [batch_size, 1, embedding_dim * 2]
        
        # Apply transformer
        transformed = self.transformer(combined)  # [batch_size, 1, embedding_dim * 2]
        transformed = transformed.squeeze(1)  # [batch_size, embedding_dim * 2]
        
        # Prediction head
        x = self.relu(self.fc1(transformed))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        rating = self.fc3(x)
        
        # Scale to rating range [0.5, 5.0]
        rating = torch.sigmoid(rating) * 4.5 + 0.5
        
        return rating.squeeze()

# ============================================================================
# Dataset class
# ============================================================================

class RatingDataset(Dataset):
    def __init__(self, df):
        self.users = torch.LongTensor(df['userId'].values)
        self.movies = torch.LongTensor(df['movieId'].values)
        self.ratings = torch.FloatTensor(df['rating'].values)
    
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]

# ============================================================================
# Training
# ============================================================================

print("\n2. Preparing data loaders...")

# Sample data for faster training (use 20% for demonstration)
# Remove this sampling for full training
SAMPLE_FRACTION = 0.2
train_sample = train_df.sample(frac=SAMPLE_FRACTION, random_state=42)
test_sample = test_df.sample(frac=SAMPLE_FRACTION, random_state=42)

print(f"Using {SAMPLE_FRACTION*100}% of data for faster training")
print(f"Sampled training set: {len(train_sample):,} ratings")
print(f"Sampled test set: {len(test_sample):,} ratings")

train_dataset = RatingDataset(train_sample)
test_dataset = RatingDataset(test_sample)

BATCH_SIZE = 2048
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"Batch size: {BATCH_SIZE}")
print(f"Training batches: {len(train_loader)}")

print("\n3. Initializing Transformer model...")

model = TransformerRecommender(
    n_users=n_users,
    n_movies=n_movies,
    embedding_dim=64,
    n_heads=4,
    n_layers=2,
    dropout=0.1
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

print("\n4. Training Transformer model...")
print("-" * 70)

N_EPOCHS = 5  # Limited epochs for demonstration
training_start = time.time()

for epoch in range(N_EPOCHS):
    model.train()
    epoch_loss = 0
    epoch_start = time.time()
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{N_EPOCHS}')
    
    for users, movies, ratings in progress_bar:
        users = users.to(device)
        movies = movies.to(device)
        ratings = ratings.to(device)
        
        # Forward pass
        predictions = model(users, movies)
        loss = criterion(predictions, ratings)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = epoch_loss / len(train_loader)
    epoch_time = time.time() - epoch_start
    
    print(f"Epoch {epoch+1}/{N_EPOCHS} - Loss: {avg_loss:.4f} - Time: {epoch_time:.2f}s")

training_time = time.time() - training_start
print("-" * 70)
print(f"✓ Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

# ============================================================================
# Evaluation
# ============================================================================

print("\n5. Evaluating on test set...")
model.eval()

all_predictions = []
all_actuals = []

prediction_start = time.time()

with torch.no_grad():
    for users, movies, ratings in tqdm(test_loader, desc='Predicting'):
        users = users.to(device)
        movies = movies.to(device)
        
        predictions = model(users, movies)
        
        all_predictions.extend(predictions.cpu().numpy())
        all_actuals.extend(ratings.numpy())

prediction_time = time.time() - prediction_start
print(f"✓ Predictions completed in {prediction_time:.2f} seconds")

all_predictions = np.array(all_predictions)
all_actuals = np.array(all_actuals)

# Calculate metrics
print("\n6. Computing evaluation metrics...")

rmse = np.sqrt(np.mean((all_actuals - all_predictions) ** 2))
mae = np.mean(np.abs(all_actuals - all_predictions))
mse = np.mean((all_actuals - all_predictions) ** 2)

ss_res = np.sum((all_actuals - all_predictions) ** 2)
ss_tot = np.sum((all_actuals - np.mean(all_actuals)) ** 2)
r2_score = 1 - (ss_res / ss_tot)

print(f"  RMSE: {rmse:.4f}")
print(f"  MAE:  {mae:.4f}")
print(f"  MSE:  {mse:.4f}")
print(f"  R²:   {r2_score:.4f}")

# ============================================================================
# Cold Start Analysis
# ============================================================================

print("\n7. Analyzing cold start performance...")

train_user_counts = train_df.groupby('userId').size()
train_movie_counts = train_df.groupby('movieId').size()

COLD_USER_THRESHOLD = 30
COLD_MOVIE_THRESHOLD = 50

test_sample['user_train_count'] = test_sample['userId'].map(train_user_counts)
test_sample['movie_train_count'] = test_sample['movieId'].map(train_movie_counts)

cold_user_mask = test_sample['user_train_count'] <= COLD_USER_THRESHOLD
cold_movie_mask = test_sample['movie_train_count'] <= COLD_MOVIE_THRESHOLD
warm_mask = ~cold_user_mask & ~cold_movie_mask

# Create prediction dataframe
results_df = test_sample.copy()
results_df['prediction'] = all_predictions

def calculate_rmse(df):
    return np.sqrt(np.mean((df['rating'] - df['prediction']) ** 2))

warm_rmse = calculate_rmse(results_df[warm_mask]) if warm_mask.sum() > 0 else None
cold_user_rmse = calculate_rmse(results_df[cold_user_mask]) if cold_user_mask.sum() > 0 else None
cold_movie_rmse = calculate_rmse(results_df[cold_movie_mask]) if cold_movie_mask.sum() > 0 else None

print(f"\nCold Start Analysis:")
print(f"  Cold user threshold: ≤{COLD_USER_THRESHOLD} ratings in train")
print(f"  Cold movie threshold: ≤{COLD_MOVIE_THRESHOLD} ratings in train")

if warm_rmse:
    print(f"\n  Warm users & movies: {warm_mask.sum():,} predictions")
    print(f"    RMSE: {warm_rmse:.4f}")

if cold_user_rmse and warm_rmse:
    print(f"\n  Cold users: {cold_user_mask.sum():,} predictions")
    print(f"    RMSE: {cold_user_rmse:.4f}")
    degradation = ((cold_user_rmse - warm_rmse) / warm_rmse) * 100
    print(f"    Degradation: +{cold_user_rmse - warm_rmse:.4f} ({degradation:.2f}%)")

if cold_movie_rmse and warm_rmse:
    print(f"\n  Cold movies: {cold_movie_mask.sum():,} predictions")
    print(f"    RMSE: {cold_movie_rmse:.4f}")
    degradation = ((cold_movie_rmse - warm_rmse) / warm_rmse) * 100
    print(f"    Degradation: +{cold_movie_rmse - warm_rmse:.4f} ({degradation:.2f}%)")

cold_start_score = ((cold_user_rmse - warm_rmse) / warm_rmse) if (warm_rmse and cold_user_rmse) else None
if cold_start_score:
    print(f"\n  Cold Start Score: {cold_start_score:.4f}")

# ============================================================================
# Precision@K
# ============================================================================

print("\n8. Computing Precision@10...")

def calculate_precision_at_k(df, k=10, threshold=3.5):
    user_groups = df.groupby('userId')
    precisions = []
    
    for user_id, group in user_groups:
        if len(group) < k:
            continue
        
        # Sort by prediction
        sorted_group = group.sort_values('prediction', ascending=False)
        top_k = sorted_group.head(k)
        
        # Count relevant items
        n_relevant = (top_k['rating'] >= threshold).sum()
        precisions.append(n_relevant / k)
    
    return np.mean(precisions) if precisions else 0.0

precision_10 = calculate_precision_at_k(results_df, k=10)
print(f"  Precision@10: {precision_10:.4f}")

# ============================================================================
# Save Results
# ============================================================================

print("\n9. Saving results...")
results_dir = '../results/'
os.makedirs(results_dir, exist_ok=True)

results = {
    'model': 'Transformer',
    'era': '2020s',
    'dataset_size': len(train_sample),
    'test_size': len(test_sample),
    'n_users': n_users,
    'n_movies': n_movies,
    'training_time_seconds': training_time,
    'prediction_time_seconds': prediction_time,
    'rmse': float(rmse),
    'mae': float(mae),
    'mse': float(mse),
    'r2_score': float(r2_score),
    'precision_at_10': float(precision_10),
    'warm_rmse': float(warm_rmse) if warm_rmse else None,
    'cold_user_rmse': float(cold_user_rmse) if cold_user_rmse else None,
    'cold_movie_rmse': float(cold_movie_rmse) if cold_movie_rmse else None,
    'cold_start_score': float(cold_start_score) if cold_start_score else None,
    'cold_user_threshold': COLD_USER_THRESHOLD,
    'cold_movie_threshold': COLD_MOVIE_THRESHOLD,
    'embedding_dim': 64,
    'n_heads': 4,
    'n_layers': 2,
    'n_epochs': N_EPOCHS,
    'batch_size': BATCH_SIZE,
    'learning_rate': 0.001,
    'sample_fraction': SAMPLE_FRACTION
}

with open(results_dir + 'transformer_results.json', 'w') as f:
    json.dump(results, f, indent=4)
print(f"  ✓ Saved: transformer_results.json")

results_df_sample = results_df[['userId', 'movieId', 'rating', 'prediction']].head(1000)
results_df_sample['error'] = results_df_sample['rating'] - results_df_sample['prediction']
results_df_sample.to_csv(results_dir + 'transformer_sample_predictions.csv', index=False)
print(f"  ✓ Saved: transformer_sample_predictions.csv")

# Save model
torch.save(model.state_dict(), results_dir + 'transformer_model.pth')
print(f"  ✓ Saved: transformer_model.pth")

print("\n" + "=" * 70)
print("TRANSFORMER MODEL EVALUATION COMPLETED!")
print("=" * 70)
print(f"\nKey Results:")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE:  {mae:.4f}")
print(f"  Precision@10: {precision_10:.4f}")
print(f"  Training time: {training_time/60:.2f} minutes")
print(f"\nResults saved in: {results_dir}")
print("\nNote: Model trained on 20% sample for demonstration.")
print("For full training, set SAMPLE_FRACTION = 1.0")
