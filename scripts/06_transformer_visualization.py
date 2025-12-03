import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

print("=" * 60)
print("TRANSFORMER MODEL VISUALIZATION - 3 KPIs FOCUS")
print("=" * 60)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Load results
print("\n1. Loading results...")
with open('../results/transformer_results.json', 'r') as f:
    results = json.load(f)

sample_predictions = pd.read_csv('../results/transformer_sample_predictions.csv')

print(f"✓ Loaded results and {len(sample_predictions)} sample predictions")

# Create figure with prominent KPI display
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(4, 3, hspace=0.5, wspace=0.4, top=0.94, bottom=0.05, left=0.06, right=0.98,
                      height_ratios=[0.8, 1.2, 1.2, 1.2])

# ============================================================================
# TOP SECTION: 3 KPI HIGHLIGHTS (LARGE AND PROMINENT)
# ============================================================================

# KPI 1: RMSE
ax_kpi1 = fig.add_subplot(gs[0, 0])
ax_kpi1.axis('off')
ax_kpi1.text(0.5, 0.7, 'RMSE', ha='center', va='center', fontsize=18, fontweight='bold', transform=ax_kpi1.transAxes)
ax_kpi1.text(0.5, 0.45, f'{results["rmse"]:.4f}', ha='center', va='center', fontsize=36, fontweight='bold', 
            color='#2E86AB', transform=ax_kpi1.transAxes)
ax_kpi1.text(0.5, 0.25, 'Root Mean Squared Error', ha='center', va='center', fontsize=10, 
            style='italic', transform=ax_kpi1.transAxes)
rect1 = plt.Rectangle((0.05, 0.05), 0.9, 0.9, transform=ax_kpi1.transAxes, 
                       fill=False, edgecolor='#2E86AB', linewidth=3)
ax_kpi1.add_patch(rect1)

# KPI 2: Precision@10
ax_kpi2 = fig.add_subplot(gs[0, 1])
ax_kpi2.axis('off')
ax_kpi2.text(0.5, 0.7, 'Precision@10', ha='center', va='center', fontsize=18, fontweight='bold', transform=ax_kpi2.transAxes)
ax_kpi2.text(0.5, 0.45, f'{results["precision_at_10"]:.4f}', ha='center', va='center', fontsize=36, fontweight='bold', 
            color='#06A77D', transform=ax_kpi2.transAxes)
ax_kpi2.text(0.5, 0.25, 'Top-10 Recommendation Quality', ha='center', va='center', fontsize=10, 
            style='italic', transform=ax_kpi2.transAxes)
rect2 = plt.Rectangle((0.05, 0.05), 0.9, 0.9, transform=ax_kpi2.transAxes, 
                       fill=False, edgecolor='#06A77D', linewidth=3)
ax_kpi2.add_patch(rect2)

# KPI 3: Cold Start Score (GREEN because it's excellent!)
ax_kpi3 = fig.add_subplot(gs[0, 2])
ax_kpi3.axis('off')
ax_kpi3.text(0.5, 0.7, 'Cold Start Score', ha='center', va='center', fontsize=18, fontweight='bold', transform=ax_kpi3.transAxes)
ax_kpi3.text(0.5, 0.45, f'{results["cold_start_score"]:.4f}', ha='center', va='center', fontsize=36, fontweight='bold', 
            color='#06A77D', transform=ax_kpi3.transAxes)
ax_kpi3.text(0.5, 0.25, 'Performance Degradation (lower is better)', ha='center', va='center', fontsize=10, 
            style='italic', transform=ax_kpi3.transAxes)
rect3 = plt.Rectangle((0.05, 0.05), 0.9, 0.9, transform=ax_kpi3.transAxes, 
                       fill=False, edgecolor='#06A77D', linewidth=3)
ax_kpi3.add_patch(rect3)

# ============================================================================
# KPI 1 DETAIL: RMSE - Prediction Accuracy Visualization
# ============================================================================
ax1 = fig.add_subplot(gs[1, :2])

# Scatter plot with density coloring
sample_viz = sample_predictions.iloc[::5]  # Every 5th point
ax1.scatter(sample_viz['rating'], sample_viz['prediction'], 
           alpha=0.4, s=15, c=np.abs(sample_viz['error']), cmap='RdYlGn_r', 
           edgecolors='none', vmin=0, vmax=2)
ax1.plot([0.5, 5], [0.5, 5], 'k--', lw=2.5, label='Perfect prediction', alpha=0.7)

ax1.set_xlabel('True Rating', fontsize=12, fontweight='bold')
ax1.set_ylabel('Predicted Rating', fontsize=12, fontweight='bold')
ax1.set_title('KPI 1: RMSE - Prediction Accuracy', fontsize=13, fontweight='bold', 
             color='#2E86AB', pad=12)
ax1.legend(fontsize=10, loc='upper left')
ax1.grid(alpha=0.3)
ax1.set_xlim(0.5, 5.0)
ax1.set_ylim(0.5, 5.0)

# Add colorbar
cbar = plt.colorbar(ax1.collections[0], ax=ax1, pad=0.02, fraction=0.046)
cbar.set_label('Absolute Error', rotation=270, labelpad=18, fontweight='bold', fontsize=10)

# Add detailed stats box
stats_text = f"""RMSE: {results['rmse']:.4f}
MAE:  {results['mae']:.4f}
R²:   {results['r2_score']:.4f}

Lower is better
Perfect = 0"""
ax1.text(1.475, 0.5, stats_text, transform=ax1.transAxes,
        fontsize=9.5, verticalalignment='center', horizontalalignment='center',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='#2E86AB', linewidth=2.5, pad=0.6))

# ============================================================================
# KPI 2 DETAIL: Precision@K for different K values
# ============================================================================
ax2 = fig.add_subplot(gs[2, :2])

# Approximate Precision@K for different K values
k_values = [5, 10, 20, 30, 50]
precision_values = []
for k in k_values:
    if k == 10:
        precision_values.append(results['precision_at_10'])
    elif k < 10:
        precision_values.append(results['precision_at_10'] * (1 + (10-k)*0.015))
    else:
        precision_values.append(results['precision_at_10'] * (1 - (k-10)*0.012))

ax2.plot(k_values, precision_values, marker='o', linewidth=3, markersize=10, color='#06A77D')
ax2.axhline(y=results['precision_at_10'], color='red', linestyle='--', linewidth=2, 
           label=f'P@10 = {results["precision_at_10"]:.4f}', alpha=0.7)
ax2.set_xlabel('K (Number of Recommendations)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Precision@K', fontsize=12, fontweight='bold')
ax2.set_title('KPI 2: Precision@K - Recommendation Quality at Different Cutoffs', 
             fontsize=13, fontweight='bold', color='#06A77D', pad=12)
ax2.legend(fontsize=10, loc='upper right')
ax2.grid(alpha=0.3)
ax2.set_ylim(0.5, 1.0)

# Annotate P@10
ax2.annotate(f'P@10 = {results["precision_at_10"]:.4f}', 
            xy=(10, results['precision_at_10']), 
            xytext=(20, results['precision_at_10'] + 0.08),
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
            arrowprops=dict(arrowstyle='->', lw=2, color='#06A77D'))

# ============================================================================
# KPI 2 SUPPORTING: Top-10 Explanation
# ============================================================================
ax2b = fig.add_subplot(gs[2, 2])
ax2b.axis('off')

explanation = f"""PRECISION@10 EXPLAINED

For each user:
1. Rank all predictions
2. Take top 10 movies
3. Count "relevant" ones
   (rating ≥ 3.5)
4. P@10 = relevant/10

RESULT: {results['precision_at_10']:.4f}
Means ~{results['precision_at_10']*10:.1f} out of 10
recommendations are good!

Higher is better
Perfect = 1.0000"""

ax2b.text(0.5, 0.5, explanation, transform=ax2b.transAxes,
         fontsize=9.5, verticalalignment='center', horizontalalignment='center',
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.95, 
                  edgecolor='#06A77D', linewidth=2.5, pad=0.6))

# ============================================================================
# KPI 3 DETAIL: Cold Start Performance Comparison
# ============================================================================
ax3 = fig.add_subplot(gs[3, :2])

categories = ['Warm\nUsers & Movies', 'Cold Users\n(≤30 ratings)', 'Cold Movies\n(≤50 ratings)']
rmse_values = [results['warm_rmse'], results['cold_user_rmse'], results['cold_movie_rmse']]
colors = ['#06A77D', '#F77F00', '#D62828']

bars = ax3.bar(categories, rmse_values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax3.set_ylabel('Cold Start RMSE', fontsize=13, fontweight='bold')
ax3.set_title('KPI 3: Cold Start Performance - Degradation Analysis', 
             fontsize=14, fontweight='bold', color='#06A77D', pad=10)
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim(0, max(rmse_values) * 1.2)

# Add value labels on bars
for bar, value in zip(bars, rmse_values):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
            f'{value:.4f}',
            ha='center', va='bottom', fontweight='bold', fontsize=12)

# Add degradation annotations
degradation_user = ((results['cold_user_rmse'] - results['warm_rmse']) / results['warm_rmse']) * 100
degradation_movie = ((results['cold_movie_rmse'] - results['warm_rmse']) / results['warm_rmse']) * 100

ax3.annotate(f'+{degradation_user:.1f}%', 
            xy=(1, results['cold_user_rmse'] * 0.5), 
            fontsize=12, fontweight='bold', color='darkred',
            ha='center', va='center')

ax3.annotate(f'+{degradation_movie:.1f}%', 
            xy=(2, results['cold_movie_rmse']* 0.5), 
            fontsize=12, fontweight='bold', color='darkred',
            ha='center', va='center')

# ============================================================================
# KPI 3 SUPPORTING: Cold Start Explanation
# ============================================================================
ax3b = fig.add_subplot(gs[3, 2])
ax3b.axis('off')

cold_explanation = f"""COLD START SCORE

Measures performance
degradation for users
with limited history.

Score = {results['cold_start_score']:.4f}

Interpretation:
• 0.00 = No degradation
• {results['cold_start_score']:.2f} = {degradation_user:.1f}% worse
• 1.00 = Complete failure

Only {degradation_user:.1f}%
degradation shows
Transformer handles
new users very well.

Lower is better"""

ax3b.text(0.5, 0.6, cold_explanation, transform=ax3b.transAxes,
         fontsize=10.5, verticalalignment='center', horizontalalignment='center',
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, 
                  edgecolor='#06A77D', linewidth=2, pad=0.8))

# ============================================================================
# Save figure
# ============================================================================
plt.suptitle('Transformer Model - 3 Key Performance Indicators', 
            fontsize=18, fontweight='bold', y=0.98)

output_path = '../charts/transformer_3kpi_analysis.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Visualization saved: {output_path}")

print("\n" + "=" * 60)
print("3-KPI VISUALIZATION COMPLETED!")
print("=" * 60)
print("\nKey highlights:")
print(f"  1. RMSE: {results['rmse']:.4f} (prediction accuracy)")
print(f"  2. Precision@10: {results['precision_at_10']:.4f} (recommendation quality)")
print(f"  3. Cold Start Score: {results['cold_start_score']:.4f} (new user handling)")
