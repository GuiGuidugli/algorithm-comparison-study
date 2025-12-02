import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate
from surprise import accuracy
import time
import json
import os

print("=" * 70)
print("MATRIX FACTORIZATION (SVD) - 2000s/2010s ERA")
print("=" * 70)

# Load processed data
print("\n1. Loading processed datasets...")
train_df = pd.read_csv('../data/processed/train.csv')
test_df = pd.read_csv('../data/processed/test.csv')

print(f"Training set: {len(train_df):,} ratings")
print(f"Test set: {len(test_df):,} ratings")
print(f"Users: {train_df['userId'].nunique():,}")
print(f"Movies: {train_df['movieId'].nunique():,}")

# Prepare data for Surprise library
print("\n2. Preparing data for Surprise library...")
reader = Reader(rating_scale=(0.5, 5.0))

# Convert to Surprise format
train_data = Dataset.load_from_df(train_df[['userId', 'movieId', 'rating']], reader)
trainset = train_data.build_full_trainset()

# Test set in Surprise format
testset = [(row['userId'], row['movieId'], row['rating']) 
           for _, row in test_df.iterrows()]

print(f"✓ Data prepared for training")

# Configure SVD model
print("\n3. Configuring SVD model...")
print("Model parameters:")
print("  - n_factors: 100 (latent factors)")
print("  - n_epochs: 20 (training iterations)")
print("  - lr_all: 0.005 (learning rate)")
print("  - reg_all: 0.02 (regularization)")

svd_model = SVD(
    n_factors=100,      # Number of latent factors
    n_epochs=20,        # Number of iterations
    lr_all=0.005,       # Learning rate
    reg_all=0.02,       # Regularization term
    random_state=42,    # For reproducibility
    verbose=True
)

# Train the model
print("\n4. Training SVD model...")
print("-" * 70)
start_time = time.time()

svd_model.fit(trainset)

training_time = time.time() - start_time
print("-" * 70)
print(f"✓ Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

# Make predictions on test set
print("\n5. Evaluating on test set...")
start_time = time.time()

predictions = svd_model.test(testset)

prediction_time = time.time() - start_time
print(f"✓ Predictions completed in {prediction_time:.2f} seconds")

# Calculate metrics
print("\n6. Computing evaluation metrics...")
rmse = accuracy.rmse(predictions, verbose=False)
mae = accuracy.mae(predictions, verbose=False)

print(f"  RMSE: {rmse:.4f}")
print(f"  MAE:  {mae:.4f}")

# Additional metrics
print("\n7. Computing additional metrics...")

# Extract predictions and actual ratings
y_true = np.array([pred.r_ui for pred in predictions])
y_pred = np.array([pred.est for pred in predictions])

# Mean Squared Error
mse = np.mean((y_true - y_pred) ** 2)

# R-squared
ss_res = np.sum((y_true - y_pred) ** 2)
ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
r2_score = 1 - (ss_res / ss_tot)

print(f"  MSE:  {mse:.4f}")
print(f"  R²:   {r2_score:.4f}")

# Analyze prediction distribution
pred_std = np.std(y_pred)
pred_range = (np.min(y_pred), np.max(y_pred))

print(f"\nPrediction statistics:")
print(f"  Mean prediction: {np.mean(y_pred):.4f}")
print(f"  Std deviation:   {pred_std:.4f}")
print(f"  Range:           {pred_range[0]:.4f} to {pred_range[1]:.4f}")

# Calculate cold start performance
print("\n8. Analyzing cold start performance...")

# Count ratings per user/movie in training set
train_user_counts = train_df.groupby('userId').size()
train_movie_counts = train_df.groupby('movieId').size()

# Define cold start thresholds
COLD_USER_THRESHOLD = 30  # Users with <= 30 ratings in train
COLD_MOVIE_THRESHOLD = 50  # Movies with <= 50 ratings in train

# Build efficient lookup dictionary from predictions
print("  Building prediction lookup...")
pred_lookup = {}
for pred in predictions:
    key = (pred.uid, pred.iid)
    pred_lookup[key] = pred

# Categorize predictions efficiently (in chunks to save memory)
print("  Categorizing predictions...")
cold_user_preds = []
cold_movie_preds = []
warm_preds = []

chunk_size = 100000
for i in range(0, len(test_df), chunk_size):
    chunk = test_df.iloc[i:i+chunk_size].copy()
    
    # Map counts
    chunk['user_train_count'] = chunk['userId'].map(train_user_counts)
    chunk['movie_train_count'] = chunk['movieId'].map(train_movie_counts)
    
    for _, row in chunk.iterrows():
        key = (row['userId'], row['movieId'])
        if key not in pred_lookup:
            continue
        
        pred = pred_lookup[key]
        user_count = row['user_train_count']
        movie_count = row['movie_train_count']
        
        if user_count <= COLD_USER_THRESHOLD:
            cold_user_preds.append(pred)
        elif movie_count <= COLD_MOVIE_THRESHOLD:
            cold_movie_preds.append(pred)
        else:
            warm_preds.append(pred)
    
    # Clear chunk from memory
    del chunk

print(f"  ✓ Categorization complete")

# Calculate RMSE for each category
from surprise import accuracy as acc

print(f"\nCold Start Analysis:")
print(f"  Cold user threshold: ≤{COLD_USER_THRESHOLD} ratings in train")
print(f"  Cold movie threshold: ≤{COLD_MOVIE_THRESHOLD} ratings in train")
print(f"\n  Warm users & movies: {len(warm_preds):,} predictions")
if len(warm_preds) > 0:
    warm_rmse = acc.rmse(warm_preds, verbose=False)
    print(f"    RMSE: {warm_rmse:.4f}")
else:
    warm_rmse = None
    print(f"    RMSE: N/A")

print(f"\n  Cold users: {len(cold_user_preds):,} predictions")
if len(cold_user_preds) > 0:
    cold_user_rmse = acc.rmse(cold_user_preds, verbose=False)
    print(f"    RMSE: {cold_user_rmse:.4f}")
    if warm_rmse:
        print(f"    Degradation: +{cold_user_rmse - warm_rmse:.4f} ({((cold_user_rmse/warm_rmse - 1) * 100):.2f}%)")
else:
    cold_user_rmse = None
    print(f"    RMSE: N/A")

print(f"\n  Cold movies: {len(cold_movie_preds):,} predictions")
if len(cold_movie_preds) > 0:
    cold_movie_rmse = acc.rmse(cold_movie_preds, verbose=False)
    print(f"    RMSE: {cold_movie_rmse:.4f}")
    if warm_rmse:
        print(f"    Degradation: +{cold_movie_rmse - warm_rmse:.4f} ({((cold_movie_rmse/warm_rmse - 1) * 100):.2f}%)")
else:
    cold_movie_rmse = None
    print(f"    RMSE: N/A")

# Overall cold start score (lower is better)
cold_start_score = None
if warm_rmse and cold_user_rmse:
    cold_start_score = (cold_user_rmse - warm_rmse) / warm_rmse
    print(f"\n  Cold Start Score: {cold_start_score:.4f}")
    print(f"    (Relative RMSE increase for cold users)")

print("\n9. Computing recommendation quality metrics...")

# Top-K recommendations metrics
def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """Calculate Precision and Recall at K"""
    
    # Group predictions by user
    user_est_true = {}
    for pred in predictions:
        user_id = pred.uid
        if user_id not in user_est_true:
            user_est_true[user_id] = []
        user_est_true[user_id].append((pred.est, pred.r_ui))
    
    precisions = []
    recalls = []
    
    for uid, user_ratings in user_est_true.items():
        # Sort by estimated rating
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        
        # Get top K
        top_k = user_ratings[:k]
        
        # Number of relevant items in top K
        n_rel_and_rec_k = sum(1 for (est, true) in top_k if true >= threshold)
        
        # Total number of relevant items
        n_rel = sum(1 for (_, true) in user_ratings if true >= threshold)
        
        # Precision@K: relevant items in top K / K
        precisions.append(n_rel_and_rec_k / k if k > 0 else 0)
        
        # Recall@K: relevant items in top K / total relevant
        recalls.append(n_rel_and_rec_k / n_rel if n_rel > 0 else 0)
    
    return np.mean(precisions), np.mean(recalls)

precision_10, recall_10 = precision_recall_at_k(predictions, k=10, threshold=3.5)

print(f"  Precision@10: {precision_10:.4f}")
print(f"  Recall@10:    {recall_10:.4f}")
print(f"  F1@10:        {2 * (precision_10 * recall_10) / (precision_10 + recall_10) if (precision_10 + recall_10) > 0 else 0:.4f}")

# Save results
print("\n10. Saving results...")
results_dir = '../results/'
os.makedirs(results_dir, exist_ok=True)

results = {
    'model': 'Matrix Factorization (SVD)',
    'era': '2000s/2010s',
    'dataset_size': len(train_df),
    'test_size': len(test_df),
    'n_users': train_df['userId'].nunique(),
    'n_movies': train_df['movieId'].nunique(),
    'training_time_seconds': training_time,
    'prediction_time_seconds': prediction_time,
    'rmse': float(rmse),
    'mae': float(mae),
    'mse': float(mse),
    'r2_score': float(r2_score),
    'precision_at_10': float(precision_10),
    'recall_at_10': float(recall_10),
    'warm_rmse': float(warm_rmse) if warm_rmse else None,
    'cold_user_rmse': float(cold_user_rmse) if cold_user_rmse else None,
    'cold_movie_rmse': float(cold_movie_rmse) if cold_movie_rmse else None,
    'cold_start_score': float(cold_start_score) if cold_start_score else None,
    'cold_user_threshold': COLD_USER_THRESHOLD,
    'cold_movie_threshold': COLD_MOVIE_THRESHOLD,
    'n_factors': 100,
    'n_epochs': 20,
    'learning_rate': 0.005,
    'regularization': 0.02
}

# Save as JSON
with open(results_dir + 'svd_results.json', 'w') as f:
    json.dump(results, f, indent=4)
print(f"  ✓ Saved: svd_results.json")

# Save as CSV for easy comparison
results_df = pd.DataFrame([results])
results_df.to_csv(results_dir + 'svd_results.csv', index=False)
print(f"  ✓ Saved: svd_results.csv")

# Save sample predictions
sample_predictions = pd.DataFrame([
    {
        'userId': pred.uid,
        'movieId': pred.iid,
        'true_rating': pred.r_ui,
        'predicted_rating': pred.est,
        'error': pred.r_ui - pred.est
    }
    for pred in predictions[:1000]  # Save first 1000 predictions as sample
])
sample_predictions.to_csv(results_dir + 'svd_sample_predictions.csv', index=False)
print(f"  ✓ Saved: svd_sample_predictions.csv (1000 samples)")

print("\n" + "=" * 70)
print("SVD MODEL EVALUATION COMPLETED!")
print("=" * 70)
print(f"\nKey Results:")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE:  {mae:.4f}")
print(f"  Training time: {training_time/60:.2f} minutes")
print(f"  Precision@10: {precision_10:.4f}")
print(f"\nResults saved in: {results_dir}")
