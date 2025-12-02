import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

print("=" * 60)
print("SVD MODEL VISUALIZATION")
print("=" * 60)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load results
print("\n1. Loading results...")
with open('../results/svd_results.json', 'r') as f:
    results = json.load(f)

sample_predictions = pd.read_csv('../results/svd_sample_predictions.csv')

print(f"✓ Loaded results and {len(sample_predictions)} sample predictions")

# Create figure with subplots
fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35, top=0.95, bottom=0.05, left=0.07, right=0.97)

# ============================================================================
# 1. Prediction vs Actual Scatter Plot
# ============================================================================
ax1 = fig.add_subplot(gs[0, :2])

# Sample for visualization (plot every 10th point to avoid overcrowding)
sample_viz = sample_predictions.iloc[::10]

ax1.scatter(sample_viz['true_rating'], sample_viz['predicted_rating'], 
           alpha=0.3, s=20, edgecolors='none')
ax1.plot([0.5, 5], [0.5, 5], 'r--', lw=2, label='Perfect prediction')
ax1.set_xlabel('True Rating', fontsize=12, fontweight='bold')
ax1.set_ylabel('Predicted Rating', fontsize=12, fontweight='bold')
ax1.set_title('SVD: Predicted vs Actual Ratings', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)
ax1.set_xlim(0.5, 5.0)
ax1.set_ylim(0.5, 5.0)

# Add RMSE annotation
ax1.text(0.05, 0.95, f"RMSE: {results['rmse']:.4f}\nMAE: {results['mae']:.4f}", 
        transform=ax1.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ============================================================================
# 2. Error Distribution
# ============================================================================
ax2 = fig.add_subplot(gs[0, 2])

errors = sample_predictions['error']
ax2.hist(errors, bins=50, edgecolor='black', alpha=0.7)
ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
ax2.set_xlabel('Prediction Error', fontsize=11, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax2.set_title('Error Distribution', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

# Add statistics
mean_error = errors.mean()
std_error = errors.std()
ax2.text(0.05, 0.95, f"Mean: {mean_error:.4f}\nStd: {std_error:.4f}", 
        transform=ax2.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

# ============================================================================
# 3. Cold Start Performance Comparison
# ============================================================================
ax3 = fig.add_subplot(gs[1, 0])

categories = ['Warm\nUsers/Movies', 'Cold\nUsers', 'Cold\nMovies']
rmse_values = [
    results['warm_rmse'],
    results['cold_user_rmse'],
    results['cold_movie_rmse']
]
colors = ['green', 'orange', 'red']

bars = ax3.bar(categories, rmse_values, color=colors, alpha=0.7, edgecolor='black')
ax3.set_ylabel('RMSE', fontsize=11, fontweight='bold')
ax3.set_title('Cold Start Performance', fontsize=12, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, rmse_values):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{value:.4f}',
            ha='center', va='bottom', fontweight='bold')

# Add degradation percentages
ax3.text(1, rmse_values[1] * 0.95, '+44.67%', ha='center', fontsize=9, color='darkred')
ax3.text(2, rmse_values[2] * 0.95, '+14.93%', ha='center', fontsize=9, color='darkred')

# ============================================================================
# 4. Rating Distribution: True vs Predicted
# ============================================================================
ax4 = fig.add_subplot(gs[1, 1])

rating_bins = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
true_hist, _ = np.histogram(sample_predictions['true_rating'], bins=rating_bins)
pred_hist, _ = np.histogram(sample_predictions['predicted_rating'], bins=rating_bins)

x = np.arange(len(rating_bins) - 1)
width = 0.35

bars1 = ax4.bar(x - width/2, true_hist, width, label='True Ratings', alpha=0.8)
bars2 = ax4.bar(x + width/2, pred_hist, width, label='Predicted Ratings', alpha=0.8)

ax4.set_xlabel('Rating', fontsize=11, fontweight='bold')
ax4.set_ylabel('Count', fontsize=11, fontweight='bold')
ax4.set_title('Rating Distribution', fontsize=12, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(['0.5', '1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0', '4.5'])
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

# ============================================================================
# 5. Metrics Summary Table
# ============================================================================
ax5 = fig.add_subplot(gs[1, 2])
ax5.axis('off')

metrics_data = [
    ['Metric', 'Value'],
    ['RMSE', f"{results['rmse']:.4f}"],
    ['MAE', f"{results['mae']:.4f}"],
    ['R²', f"{results['r2_score']:.4f}"],
    ['', ''],
    ['Precision@10', f"{results['precision_at_10']:.4f}"],
    ['Recall@10', f"{results['recall_at_10']:.4f}"],
    ['', ''],
    ['Training Time', f"{results['training_time_seconds']/60:.2f} min"],
    ['Prediction Time', f"{results['prediction_time_seconds']:.2f} sec"],
]

table = ax5.table(cellText=metrics_data, cellLoc='left', loc='center',
                 colWidths=[0.6, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)

# Style header
for i in range(2):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style separator rows
table[(4, 0)].set_facecolor('#E0E0E0')
table[(4, 1)].set_facecolor('#E0E0E0')
table[(7, 0)].set_facecolor('#E0E0E0')
table[(7, 1)].set_facecolor('#E0E0E0')

ax5.set_title('Model Performance Summary', fontsize=12, fontweight='bold', pad=20)

# ============================================================================
# 6. Absolute Error by True Rating
# ============================================================================
ax6 = fig.add_subplot(gs[2, 0])

sample_predictions['abs_error'] = sample_predictions['error'].abs()
rating_groups = sample_predictions.groupby('true_rating')['abs_error'].mean()

ax6.plot(rating_groups.index, rating_groups.values, marker='o', linewidth=2, markersize=8)
ax6.set_xlabel('True Rating', fontsize=11, fontweight='bold')
ax6.set_ylabel('Mean Absolute Error', fontsize=11, fontweight='bold')
ax6.set_title('Error by Rating Value', fontsize=12, fontweight='bold')
ax6.grid(alpha=0.3)
ax6.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])

# ============================================================================
# 7. Model Architecture Diagram (Text-based)
# ============================================================================
ax7 = fig.add_subplot(gs[2, 1:])
ax7.axis('off')

arch_text = f"""SVD MODEL ARCHITECTURE

User-Movie Rating Matrix (Sparse)
       ↓
Matrix Factorization (SVD)
       ↓
User Matrix (U)  ×  Item Matrix (V^T)
[85,307 × 100]      [100 × 13,063]
       ↓
Predicted Rating = U[user] · V[movie]

HYPERPARAMETERS:
• Latent Factors: {results['n_factors']}
• Epochs: {results['n_epochs']}
• Learning Rate: {results['learning_rate']}
• Regularization: {results['regularization']}

COLD START HANDLING:
• Cold User Threshold: ≤{results['cold_user_threshold']} ratings
• Cold Movie Threshold: ≤{results['cold_movie_threshold']} ratings
• Cold Start Score: {results['cold_start_score']:.4f}
  (0 = no degradation, 1 = complete failure)"""

ax7.text(0.5, 0.5, arch_text, transform=ax7.transAxes,
        fontsize=10, verticalalignment='center', horizontalalignment='center',
        family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, pad=0.8))

# ============================================================================
# Save figure
# ============================================================================
plt.suptitle('Matrix Factorization (SVD) - Performance Analysis', 
            fontsize=16, fontweight='bold', y=0.995)

output_path = '../charts/svd_performance_analysis.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Visualization saved: {output_path}")

print("\n" + "=" * 60)
print("VISUALIZATION COMPLETED!")
print("=" * 60)