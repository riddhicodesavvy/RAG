"""Generate visualizations for Enterprise RAG benchmark results."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_JSON = ROOT / "benchmarks" / "results" / "rag_results.json"
FIG_DIR = ROOT / "benchmarks" / "figures"


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 180})

    # 1. Experiment A: Recall@1, Recall@3, MRR by Question Category
    exp_a = data["experiment_a_baseline_1000_200"]
    cats = ["direct_factual", "procedural", "paraphrased", "exact_id", "ambiguous"]
    r1 = [exp_a["by_category"][c]["recall_at_1"] for c in cats]
    r3 = [exp_a["by_category"][c]["recall_at_3"] for c in cats]
    mrr = [exp_a["by_category"][c]["mrr"] for c in cats]

    x = np.arange(len(cats))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.bar(x - w, r1, w, label="Recall@1")
    ax.bar(x, r3, w, label="Recall@3")
    ax.bar(x + w, mrr, w, label="MRR")
    ax.set_xticks(x)
    ax.set_xticklabels(["Direct Factual\n(n=25)", "Procedural\n(n=20)", "Paraphrased\n(n=20)", "Exact Tech ID\n(n=15)", "Ambiguous\n(n=10)"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Enterprise RAG Retrieval Quality by Query Category (1000/200 Baseline)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rag_exp_a_by_category.png", bbox_inches="tight")
    plt.close(fig)

    # 2. Experiment B + C: Chunking Ablation & Confidence Gate Sweep
    exp_b = data["experiment_b_chunking_ablation"]
    cfgs = [f"{r['index_stats']['chunk_size']}/{r['index_stats']['chunk_overlap']}" for r in exp_b]
    b_r1 = [r["retrieval"]["overall_answerable"]["recall_at_1"] for r in exp_b]
    b_r3 = [r["retrieval"]["overall_answerable"]["recall_at_3"] for r in exp_b]
    b_mrr = [r["retrieval"]["overall_answerable"]["mrr"] for r in exp_b]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    xb = np.arange(len(cfgs))
    axes[0].bar(xb - 0.25, b_r1, 0.25, label="Recall@1")
    axes[0].bar(xb, b_r3, 0.25, label="Recall@3")
    axes[0].bar(xb + 0.25, b_mrr, 0.25, label="MRR")
    axes[0].set_xticks(xb)
    axes[0].set_xticklabels([f"{c}\n({r['index_stats']['num_chunks']} chunks)" for c, r in zip(cfgs, exp_b)])
    axes[0].set_ylim(0, 1.08)
    axes[0].set_title("Chunking Ablation (n=90 Answerable Queries)")
    axes[0].set_ylabel("Score")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend()

    sweep = data["experiment_c_confidence_gating"]["threshold_sweep"]
    taus = [r["distance_threshold_tau"] for r in sweep]
    acc_valid = [r["answerable_acceptance_rate"] * 100 for r in sweep]
    rej_oos = [r["correct_rejection_rate_oos"] * 100 for r in sweep]
    axes[1].plot(taus, acc_valid, "o-", label="Valid Query Acceptance (%)", linewidth=2)
    axes[1].plot(taus, rej_oos, "s--", label="Out-of-Scope Rejection (%)", linewidth=2)
    opt = data["experiment_c_confidence_gating"]["optimal_youden_threshold"]
    axes[1].axvline(opt, color="#d62728", linestyle=":", label=f"Optimal tau={opt}")
    axes[1].set_title("Confidence Distance-Gate Calibration (L2 Space)")
    axes[1].set_xlabel("Squared L2 Distance Threshold (tau)")
    axes[1].set_ylabel("Rate (%)")
    axes[1].set_ylim(-5, 108)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "rag_exp_bc_ablation_gating.png", bbox_inches="tight")
    plt.close(fig)
    print("Generated RAG figures in:", FIG_DIR)


if __name__ == "__main__":
    main()

