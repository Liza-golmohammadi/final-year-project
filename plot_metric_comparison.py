import matplotlib.pyplot as plt
import numpy as np

conditions = ['MIT-only', 'Balanced Pooled']
svm_gain = [10.5, 16.5]
rf_gain  = [10.3, 17.9]

x = np.arange(len(conditions))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 6))

bars1 = ax.bar(x - width/2, svm_gain, width, label='SVM',
               color='#029E73', edgecolor='black', linewidth=1)
bars2 = ax.bar(x + width/2, rf_gain, width, label='Random Forest',
               color='#DE8F05', edgecolor='black', linewidth=1)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'+{height:.1f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, label='Baseline (ΔAUC = 0)')
ax.set_ylabel('AUC Improvement Over Baseline (points)', fontsize=12, fontweight='bold')
ax.set_xlabel('Experimental Condition', fontsize=12, fontweight='bold')
ax.set_title('Machine Learning Value: Gain Over Simple Baseline',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(conditions, fontsize=11)
ax.legend(loc='upper right', fontsize=10, frameon=True)
ax.set_ylim([0, 22])
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('figures/figure_baseline_delta.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved: figure_baseline_delta.png")
