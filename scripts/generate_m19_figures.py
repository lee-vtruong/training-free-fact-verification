"""Generates high-resolution visualization figures for Milestone M1.9."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def generate_figures(output_dir: Path):
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Macro-F1 and Accuracy by Arm
    metrics_csv = output_dir / "metrics_by_arm.csv"
    if metrics_csv.exists():
        arms = []
        macro_f1 = []
        acc = []
        with open(metrics_csv, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
            header = lines[0].split(",")
            for line in lines[1:]:
                parts = line.split(",")
                arms.append(parts[0])
                macro_f1.append(float(parts[1]))
                acc.append(float(parts[2]))

        x = np.arange(len(arms))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        rects1 = ax.bar(x - width/2, macro_f1, width, label='Macro-F1', color='#2563EB')
        rects2 = ax.bar(x + width/2, acc, width, label='Accuracy', color='#10B981')

        ax.set_ylabel('Score')
        ax.set_title('M1.9 Performance Across 6 Experimental Arms (Qwen3-8B BF16)')
        ax.set_xticks(x)
        ax.set_xticklabels(arms, fontweight='bold')
        ax.set_ylim(0, max(max(acc), max(macro_f1)) * 1.3)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f'{h:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f'{h:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

        fig.tight_layout()
        fig.savefig(figures_dir / "macro_f1_and_acc_by_arm.png")
        plt.close(fig)

    # 2. Hypothesis Comparisons & 95% Bootstrap CIs
    stats_json = output_dir / "statistical_tests.json"
    if stats_json.exists():
        with open(stats_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        tests = data.get("hypotheses_evaluated", [])
        if tests:
            comps = [t["comparison"] for t in tests]
            deltas = [t["observed_delta_macro_f1"] for t in tests]
            ci_low = [t["macro_f1_95_ci"][0] for t in tests]
            ci_high = [t["macro_f1_95_ci"][1] for t in tests]
            yerr = [
                [deltas[i] - ci_low[i] for i in range(len(tests))],
                [ci_high[i] - deltas[i] for i in range(len(tests))]
            ]

            fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
            y_pos = np.arange(len(comps))
            ax.errorbar(deltas, y_pos, xerr=yerr, fmt='o', color='#DC2626',
                        ecolor='#4B5563', elinewidth=2, capsize=5, markersize=7)
            ax.axvline(0, color='gray', linestyle='--', linewidth=1)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(comps, fontweight='bold')
            ax.set_xlabel(r'$\Delta$ Macro-F1 (with 95% Paired Bootstrap CI)')
            ax.set_title('Hypothesis Battery: Effect Sizes and Confidence Intervals')
            ax.grid(axis='x', linestyle='--', alpha=0.5)

            fig.tight_layout()
            fig.savefig(figures_dir / "hypothesis_bootstrap_cis.png")
            plt.close(fig)

    # 3. Confusion Matrices
    cm_json = output_dir / "confusion_matrices.json"
    if cm_json.exists():
        with open(cm_json, "r", encoding="utf-8") as f:
            cms = json.load(f)

        classes = ["Supported", "Refuted", "Not Enough Info", "Conflicting"]
        short_classes = ["SUP", "REF", "NEI", "CF"]
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=300)
        axes = axes.flatten()

        for idx, (arm, matrix) in enumerate(cms.items()):
            if idx >= len(axes):
                break
            ax = axes[idx]
            arr = np.array([[matrix.get(g, {}).get(p, 0) for p in classes] for g in classes])
            im = ax.imshow(arr, cmap='Blues', vmin=0, vmax=max(np.max(arr), 1))

            ax.set_title(f"Arm {arm}", fontweight='bold')
            ax.set_xticks(np.arange(len(classes)))
            ax.set_yticks(np.arange(len(classes)))
            ax.set_xticklabels(short_classes)
            ax.set_yticklabels(short_classes)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Gold")

            for i in range(len(classes)):
                for j in range(len(classes)):
                    val = arr[i, j]
                    ax.text(j, i, str(val), ha='center', va='center',
                            color="white" if val > np.max(arr)/2 and val > 0 else "black")

        fig.tight_layout()
        fig.savefig(figures_dir / "confusion_matrices_grid.png")
        plt.close(fig)

    print(f"Generated all figures in {figures_dir}")

if __name__ == "__main__":
    generate_figures(Path("outputs/m19-historical-outcome-eval-v1"))
