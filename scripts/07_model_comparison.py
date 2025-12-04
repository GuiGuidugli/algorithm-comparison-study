import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

print("=" * 70)
print("MODEL COMPARISON: SVD vs TRANSFORMER")
print("=" * 70)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load results from both models
print("\n1. Loading results...")
with open('../results/svd_results.json', 'r') as f:
    svd_results = json.load(f)

with open('../results/transformer_results.json', 'r') as f:
    transformer_results = json.load(f)

print("✓ Loaded SVD and Transformer results")

# Create comprehensive comparison figure
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35, top=0.93, bottom=0.06, left=0.06, right=0.97)

# ============================================================================
# 1. RMSE Comparison (Bar Chart)
# ============================================================================
ax1 = fig.add_subplot(gs[0, 0])

models = ['SVD\n(2000s)', 'Transformer\n(2020s)']
rmse_values = [svd_results['rmse'], transformer_results['rmse']]
colors = ['#3498db', '#e74c3c']

bars = ax1.bar(models, rmse_values, color=colors, alpha=0.8, edgecolor='black', linewidth=2, width=0.6)
ax1.set_ylabel('RMSE', fontsize=13, fontweight='bold')
ax1.set_title('KPI 1: RMSE Comparison', fontsize=14, fontweight='bold', pad=12)
ax1.set_ylim(0, max(rmse_values) * 1.3)
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for bar, value in zip(bars, rmse_values):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{value:.4f}',
            ha='center', va='bottom', fontweight='bold', fontsize=12)

# Add winner indicator
winner_idx = np.argmin(rmse_values)
ax1.text(winner_idx, rmse_values[winner_idx] * 0.5, '✓ WINNER',
        ha='center', va='center', fontsize=11, fontweight='bold',
        color='white', bbox=dict(boxstyle='round', facecolor='green', alpha=0.8))

difference = abs(rmse_values[0] - rmse_values[1])
ax1.text(0.5, 0.95, f'Δ = {difference:.4f}', transform=ax1.transAxes,
        ha='center', va='top', fontsize=10, style='italic')

# ============================================================================
# 2. Precision@10 Comparison (Bar Chart)
# ============================================================================
ax2 = fig.add_subplot(gs[0, 1])

precision_values = [svd_results['precision_at_10'], transformer_results['precision_at_10']]

bars = ax2.bar(models, precision_values, color=colors, alpha=0.8, edgecolor='black', linewidth=2, width=0.6)
ax2.set_ylabel('Precision@10', fontsize=13, fontweight='bold')
ax2.set_title('KPI 2: Precision@10 Comparison', fontsize=14, fontweight='bold', pad=12)
ax2.set_ylim(0, 1.0)
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bar, value in zip(bars, precision_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{value:.4f}',
            ha='center', va='bottom', fontweight='bold', fontsize=12)

# Add winner indicator
winner_idx = np.argmax(precision_values)
ax2.text(winner_idx, precision_values[winner_idx] * 0.5, '✓ WINNER',
        ha='center', va='center', fontsize=11, fontweight='bold',
        color='white', bbox=dict(boxstyle='round', facecolor='green', alpha=0.8))

difference = abs(precision_values[0] - precision_values[1])
improvement = (difference / precision_values[0]) * 100
ax2.text(0.5, 0.95, f'Δ = {difference:.4f}\n(+{improvement:.1f}%)', transform=ax2.transAxes,
        ha='center', va='top', fontsize=10, style='italic')

# ============================================================================
# 3. Cold Start Score Comparison (Bar Chart)
# ============================================================================
ax3 = fig.add_subplot(gs[0, 2])

cold_start_values = [svd_results['cold_start_score'], transformer_results['cold_start_score']]

bars = ax3.bar(models, cold_start_values, color=colors, alpha=0.8, edgecolor='black', linewidth=2, width=0.6)
ax3.set_ylabel('Cold Start Score', fontsize=13, fontweight='bold')
ax3.set_title('KPI 3: Cold Start Score Comparison', fontsize=14, fontweight='bold', pad=12)
ax3.set_ylim(0, max(cold_start_values) * 1.3)
ax3.grid(axis='y', alpha=0.3)

# Add value labels
for bar, value in zip(bars, cold_start_values):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{value:.4f}',
            ha='center', va='bottom', fontweight='bold', fontsize=12)

# Add winner indicator (lower is better for cold start)
winner_idx = np.argmin(cold_start_values)
ax3.text(winner_idx, cold_start_values[winner_idx] * 0.5, '✓ WINNER',
        ha='center', va='center', fontsize=11, fontweight='bold',
        color='white', bbox=dict(boxstyle='round', facecolor='green', alpha=0.8))

difference = abs(cold_start_values[0] - cold_start_values[1])
improvement = (difference / cold_start_values[0]) * 100
ax3.text(0.5, 0.95, f'Δ = {difference:.4f}\n(-{improvement:.1f}%)', transform=ax3.transAxes,
        ha='center', va='top', fontsize=10, style='italic')

# ============================================================================
# 4. Cold Start Detailed Comparison (Grouped Bar Chart)
# ============================================================================
ax4 = fig.add_subplot(gs[1, :])

categories = ['Warm\nUsers & Movies', 'Cold Users\n(≤30 ratings)', 'Cold Movies\n(≤50 ratings)']
svd_cold_rmse = [svd_results['warm_rmse'], svd_results['cold_user_rmse'], svd_results['cold_movie_rmse']]
transformer_cold_rmse = [transformer_results['warm_rmse'], transformer_results['cold_user_rmse'], transformer_results['cold_movie_rmse']]

x = np.arange(len(categories))
width = 0.35

bars1 = ax4.bar(x - width/2, svd_cold_rmse, width, label='SVD (2000s)', 
               color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax4.bar(x + width/2, transformer_cold_rmse, width, label='Transformer (2020s)', 
               color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)

ax4.set_ylabel('RMSE', fontsize=13, fontweight='bold')
#ax4.set_xlabel('User/Movie Category', fontsize=13, fontweight='bold')
ax4.set_title('Cold Start Performance Comparison - Detailed Breakdown', fontsize=15, fontweight='bold', pad=15)
ax4.set_xticks(x)
ax4.set_xticklabels(categories, fontsize=11)
ax4.legend(fontsize=12, loc='upper left')
ax4.grid(axis='y', alpha=0.3)
ax4.set_ylim(0, max(max(svd_cold_rmse), max(transformer_cold_rmse)) * 1.15)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# ============================================================================
# 5. Training Time Comparison
# ============================================================================
ax5 = fig.add_subplot(gs[2, 1])

training_times = [
    svd_results['training_time_seconds'] / 60,
    transformer_results['training_time_seconds'] / 60
]

bars = ax5.bar(models, training_times, color=colors, alpha=0.8, edgecolor='black', linewidth=2, width=0.6)
ax5.set_ylabel('Training Time (minutes)', fontsize=13, fontweight='bold')
ax5.set_title('Training Time Comparison', fontsize=14, fontweight='bold', pad=12)
ax5.grid(axis='y', alpha=0.3)

# Add value labels
for bar, value in zip(bars, training_times):
    height = bar.get_height()
    height_offset = 0.2
    ax5.text(bar.get_x() + bar.get_width()/2., height + height_offset,
            f'{value:.2f} min',
            ha='center', va='bottom', fontweight='bold', fontsize=11)

# Add note
speedup = training_times[1] / training_times[0]
ax5.text(0.5, 0.95, f'Transformer is {speedup:.1f}x slower', transform=ax5.transAxes,
        ha='right', va='center', fontsize=10, style='italic', color='red')

# ============================================================================
# 6. All Metrics Summary Table
# ============================================================================
ax6 = fig.add_subplot(gs[2, 1:])
ax6.axis('off')

# Create summary data
#summary_data = [
#    ['Metric', 'SVD (2000s)', 'Transformer (2020s)', 'Winner'],
#    ['', '', '', ''],
#    ['RMSE ↓', f"{svd_results['rmse']:.4f}", f"{transformer_results['rmse']:.4f}", 
#     'Transformer' if transformer_results['rmse'] < svd_results['rmse'] else 'SVD'],
#    ['MAE ↓', f"{svd_results['mae']:.4f}", f"{transformer_results['mae']:.4f}",
#     'Transformer' if transformer_results['mae'] < svd_results['mae'] else 'SVD'],
#    ['R² ↑', f"{svd_results['r2_score']:.4f}", f"{transformer_results['r2_score']:.4f}",
#     'Transformer' if transformer_results['r2_score'] > svd_results['r2_score'] else 'SVD'],
#    ['', '', '', ''],
#    ['Precision@10 ↑', f"{svd_results['precision_at_10']:.4f}", f"{transformer_results['precision_at_10']:.4f}",
#     'Transformer' if transformer_results['precision_at_10'] > svd_results['precision_at_10'] else #'SVD'],
#    ['Recall@10 ↑', f"{svd_results['recall_at_10']:.4f}", f"{transformer_results.get('recall_at_10', 'N/#A')}", 'SVD'],
#    ['', '', '', ''],
#    ['Cold Start Score ↓', f"{svd_results['cold_start_score']:.4f}", #f"{transformer_results['cold_start_score']:.4f}",
#     'Transformer' if transformer_results['cold_start_score'] < svd_results['cold_start_score'] else #'SVD'],
#    ['Warm RMSE', f"{svd_results['warm_rmse']:.4f}", f"{transformer_results['warm_rmse']:.4f}", '-'],
#    ['Cold User RMSE', f"{svd_results['cold_user_rmse']:.4f}", f"{transformer_results['cold_user_rmse']:.4f}", '-'],
#    ['', '', '', ''],
#    ['Training Time', f"{svd_results['training_time_seconds']/60:.2f} min", 
#     f"{transformer_results['training_time_seconds']/60:.2f} min", 'SVD'],
#    ['Model Parameters', '~2.5M', '~2.0M', 'Transformer'],
#]

#table = ax6.table(cellText=summary_data, cellLoc='center', loc='center',
#                 colWidths=[0.3, 0.25, 0.25, 0.2])
#table.auto_set_font_size(False)
#table.set_fontsize(10)
#table.scale(1, 2.2)

# Style header
#for i in range(4):
#    cell = table[(0, i)]
#    cell.set_facecolor('#2C3E50')
#    cell.set_text_props(weight='bold', color='white', fontsize=11)

# Style separator rows
#for row_idx in [1, 5, 8, 12]:
#    for col_idx in range(4):
#        table[(row_idx, col_idx)].set_facecolor('#ECF0F1')

# Highlight winners in green
#for row_idx in range(len(summary_data)):
#    if row_idx > 1 and summary_data[row_idx][3] in ['SVD', 'Transformer']:
#        winner = summary_data[row_idx][3]
#        col_idx = 1 if winner == 'SVD' else 2
#        table[(row_idx, col_idx)].set_facecolor('#D5F4E6')
#        table[(row_idx, 3)].set_facecolor('#52BE80')
#        table[(row_idx, 3)].set_text_props(weight='bold', color='white')

#ax6.set_title('Complete Performance Summary', fontsize=15, fontweight='bold', pad=20)

# ============================================================================
# Overall Title and Summary
# ============================================================================
plt.suptitle('Recommendation Systems Evolution: SVD (2000s) vs Transformer (2020s)', 
            fontsize=20, fontweight='bold', y=0.98)

# ============================================================================
# Save figure
# ============================================================================
output_path = '../charts/model_comparison_svd_vs_transformer.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Comparison visualization saved: {output_path}")

# Also save comparison data as CSV
comparison_df = pd.DataFrame({
    'Metric': ['RMSE', 'MAE', 'R²', 'Precision@10', 'Cold Start Score', 'Training Time (min)'],
    'SVD_2000s': [
        svd_results['rmse'],
        svd_results['mae'],
        svd_results['r2_score'],
        svd_results['precision_at_10'],
        svd_results['cold_start_score'],
        svd_results['training_time_seconds'] / 60
    ],
    'Transformer_2020s': [
        transformer_results['rmse'],
        transformer_results['mae'],
        transformer_results['r2_score'],
        transformer_results['precision_at_10'],
        transformer_results['cold_start_score'],
        transformer_results['training_time_seconds'] / 60
    ]
})

comparison_df['Winner'] = comparison_df.apply(
    lambda row: 'Transformer' if (
        (row.name in [0, 1, 4, 5] and row['Transformer_2020s'] < row['SVD_2000s']) or
        (row.name in [2, 3] and row['Transformer_2020s'] > row['SVD_2000s'])
    ) else 'SVD',
    axis=1
)

comparison_df.to_csv('../results/model_comparison.csv', index=False)
print(f"✓ Comparison data saved: ../results/model_comparison.csv")

print("\n" + "=" * 70)
print("MODEL COMPARISON COMPLETED!")
print("=" * 70)

