import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ==============================
# RF Balanced Pooled — Aggregated Confusion Matrix
# TP=2858, FP=1144, TN=1354, FN=878
# ==============================
cm = np.array([[1354, 1144],
               [878,  2858]])

labels = ['HC (Control)', 'PD']

fig, ax = plt.subplots(figsize=(6, 5))

# Colours: correct = green shades, errors = red shades
colors = np.array([['#2ecc71', '#e74c3c'],
                   ['#e74c3c', '#2ecc71']])

for i in range(2):
    for j in range(2):
        ax.add_patch(plt.Rectangle((j, 1-i), 1, 1, color=colors[i][j], alpha=0.7))
        ax.text(j + 0.5, 1 - i + 0.5, str(cm[i][j]),
                ha='center', va='center', fontsize=18, fontweight='bold', color='white')

ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
ax.set_xticks([0.5, 1.5])
ax.set_yticks([0.5, 1.5])
ax.set_xticklabels(labels, fontsize=12)
ax.set_yticklabels(['PD', 'HC (Control)'], fontsize=12)
ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax.set_title('Confusion Matrix — RF Balanced Pooled\n(Window-level, threshold=0.5, aggregated 5-fold)',
             fontsize=12, fontweight='bold')

# Add TN/FP/FN/TP labels
ax.text(0.5, 1.5, 'TN', ha='center', va='top', fontsize=9, color='white', style='italic')
ax.text(1.5, 1.5, 'FP', ha='center', va='top', fontsize=9, color='white', style='italic')
ax.text(0.5, 0.5, 'FN', ha='center', va='top', fontsize=9, color='white', style='italic')
ax.text(1.5, 0.5, 'TP', ha='center', va='top', fontsize=9, color='white', style='italic')

plt.tight_layout()
plt.savefig('figures/figure_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: figure_confusion_matrix.png")
print(f"\nSensitivity (Recall): {2858/(2858+878):.3f}")
print(f"Specificity:          {1354/(1354+1144):.3f}")
print(f"F1-score (approx):    {2*2858/(2*2858+1144+878):.3f}")
