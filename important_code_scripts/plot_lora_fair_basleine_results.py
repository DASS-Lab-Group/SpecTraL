"""Generate final plots after training"""

import json
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
result_file = 'LoRA-FAIR/results/lora_fair_nicopp_1_100c_50r/results_lora_fair_nicopp.json'
with open(result_file) as f:
    data = json.load(f)

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Loss curve
ax1 = axes[0]
if 'round_losses' in data:
    ax1.plot(data['round'], data['round_losses'], 'b-o', linewidth=2, markersize=5)
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Average Loss', fontsize=12)
    ax1.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

# Accuracy curve
ax2 = axes[1]
if 'avg_top1' in data:
    ax2.plot(data['round'], data['avg_top1'], 'g-o', linewidth=2, markersize=5, label='Top-1')
    if 'avg_top5' in data:
        ax2.plot(data['round'], data['avg_top5'], 'r-s', linewidth=2, markersize=5, label='Top-5')
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Test Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/lora_fair_plots/lora_fair_training_curves_1.png', dpi=150, bbox_inches='tight')
print("✓ Saved to results/training_curves.png")
plt.show()