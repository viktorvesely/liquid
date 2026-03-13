import matplotlib.pyplot as plt

def plot_total_loss(results, exp_dir):
    h_body = [r["h_body"] for r in results]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(h_body, [r["best_val_loss"] for r in results], "o-", label="Best val", color="black")
    ax.plot(h_body, [r["final_train_loss"] for r in results], "s--", label="Final train", color="gray")
    ax.set_xlabel("Body hidden width")
    ax.set_ylabel("Total loss")
    ax.set_title("Total loss (CE + auxiliary)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{exp_dir}/plots/total_loss.png", dpi=150)
    plt.close()


def plot_ce_loss(results, exp_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in results:
        ax.plot(r["metrics"]["validation_ce_loss"], label=f"h_body={r['h_body']}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CE loss")
    ax.set_title("Cross-entropy loss (validation)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{exp_dir}/plots/ce_loss.png", dpi=150)
    plt.close()


def plot_specialization_loss(results, exp_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in results:
        key = "validation_specialization_losss"
        if key in r["metrics"]:
            ax.plot(r["metrics"][key], label=f"h_body={r['h_body']}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Specialization loss")
    ax.set_title("Specialization loss (validation)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{exp_dir}/plots/specialization_loss.png", dpi=150)
    plt.close()


def plot_load_distribution_loss(results, exp_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in results:
        key = "validation_load_distribution_loss"
        if key in r["metrics"]:
            ax.plot(r["metrics"][key], label=f"h_body={r['h_body']}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Load distribution loss")
    ax.set_title("Load distribution loss (validation)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{exp_dir}/plots/load_distribution_loss.png", dpi=150)
    plt.close()


def plot_ablation(results, exp_dir):
    """Generate all ablation plots."""
    plot_total_loss(results, exp_dir)
    plot_ce_loss(results, exp_dir)
    plot_specialization_loss(results, exp_dir)
    plot_load_distribution_loss(results, exp_dir)