"""Real-time training monitor"""

import json
import time
import matplotlib.pyplot as plt
from pathlib import Path

result_file = Path('results/lora_fair_feature_noniid_100c_75r/results_lora_fair_nicopp.json')

plt.ion()  # Interactive mode
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

print("Monitoring training... Press Ctrl+C to stop")
print(f"Watching: {result_file}")

try:
    while True:
        if result_file.exists():
            print("result file exists")
            with open(result_file) as f:
                data = json.load(f)
            
            # Clear axes
            ax1.clear()
            ax2.clear()
            
            # Plot Loss
            if 'round_losses' in data and data['round_losses']:
                rounds = data['round']
                losses = data['round_losses']
                ax1.plot(rounds, losses, 'b-o', linewidth=2, markersize=4)
                ax1.set_xlabel('Round', fontsize=11)
                ax1.set_ylabel('Loss', fontsize=11)
                ax1.set_title(f'Training Loss (Current: {losses[-1]:.4f})', fontsize=12, fontweight='bold')
                ax1.grid(True, alpha=0.3)
            
            # Plot Accuracy
            if 'avg_top1' in data and data['avg_top1']:
                acc_rounds = data['round']
                accs = data['avg_top1']
                ax2.plot(acc_rounds, accs, 'g-o', linewidth=2, markersize=4)
                ax2.set_xlabel('Round', fontsize=11)
                ax2.set_ylabel('Accuracy (%)', fontsize=11)
                ax2.set_title(f'Test Accuracy (Current: {accs[-1]:.2f}%)', fontsize=12, fontweight='bold')
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.pause(10)  # Update every 10 seconds
        else:
            print(f"Waiting for {result_file}...")
            time.sleep(10)
            
except KeyboardInterrupt:
    print("\n✓ Monitoring stopped")
    plt.close()