import matplotlib.pyplot as plt
import numpy as np

# Set overall font style to match academic papers
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

# 1. LOEO Summary Plot (Fig. 2)
def plot_loeo_summary():
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    sources = ['Held-out EyePACS', 'Held-out DDR', 'Held-out APTOS']
    qwk_values = [0.524, 0.787, 0.863]
    
    bars = ax.bar(sources, qwk_values, color='#459e67', width=0.65)
    ax.axhline(y=0.813, color='#b73838', linestyle='--', linewidth=1.5, label='in-domain ceiling (0.813)')
    
    ax.set_ylabel('QWK on held-out source', fontsize=11)
    ax.set_title('Cross-environment generalisation (LOEO)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', frameon=True)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
                    
    plt.tight_layout()
    plt.savefig('loeo_summary.png')
    plt.close()

# 2. Ablation Study Bars
def plot_ablation():
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    configs = ['Full teacher', 'No-casual (Mod.1)', 'Label-free (Mod.3)']
    qwk_values = [0.813, 0.861, 0.839]
    colors = ['#4c72b0', '#dd8452', '#55a868']
    
    bars = ax.bar(configs, qwk_values, color=colors, width=0.6)
    ax.set_ylabel('test QWK', fontsize=11)
    ax.set_title('Ablation: contribution of each module', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.0)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
                    
    plt.tight_layout()
    plt.savefig('ablation_bars.png')
    plt.close()

# 3. Student vs Teacher Distillation Trade-off
def plot_student_vs_teacher():
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    models = ['Teacher', 'Student']
    accuracy = [0.829, 0.851]
    qwk = [0.813, 0.840]
    
    x = np.arange(len(models))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, accuracy, width, label='Accuracy', color='#4c72b0')
    rects2 = ax.bar(x + width/2, qwk, width, label='QWK', color='#dd8452')
    
    ax.set_ylabel('', fontsize=11)
    ax.set_title('Distillation trade-off', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', fontsize=10, frameon=True)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
                        
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig('student_vs_teacher.png')
    plt.close()

if __name__ == '__main__':
    plot_loeo_summary()
    plot_ablation()
    plot_student_vs_teacher()
    print("All primary manuscript figures generated successfully!")
