import pandas as pd
max_lp = 4

samples = [
    ("video", "mamba",  "VideoMamba", 26.8, 64.2),
    ("video", "mamba",  "OtherMamba", 24.2, 71.4),
    ("video", "CNN", "SlowFast", 62.8, 63.1),
    ("video", "transformer", "VideoMAE", 87.3, 74.5),
    ("video", "transformer", "Swin-B", 88.1, 69.6),
    ("video", "transformer", "UniFormer-S", 21.4, 69.4),
    ("video", "CNN", "TSM", 24.3, 61.7),
    ("language", "mamba", "MoE-Mamba", 117, 1 - (2.81 / max_lp)),
    ("language", "transformer", "MoE-Transformer", 114, 1 - (2.88 / max_lp)),
    ("language", "transformer", "Megatron-gpt2", 345, 1 - (2.960 / max_lp)),
    ("point", "transformer", "Point\nTransformer", 4.9, 93.7),
    ("point", "transformer", "Point-M2AE", 12.8, 93.4),
    ("point", "transformer", "PointGPT", 12, 94.9),
    ("point", "mamba", "PointMamba", 12.3, 93.6),
    ("point", "CNN", "PointNeXt-S ", 1.4, 93.2),
    ("point", "CNN", "KPConv ", 15, 92.9),
]

df = pd.DataFrame(samples, columns=["domain", "type", "name", "params", "score"])

def quantile_transform(group):
    group["param_quantile"] = group["params"].rank(pct=True)
    group["score_quantile"] = group["score"].rank(pct=True)
    return group

df = df.groupby("domain", group_keys=False).apply(quantile_transform)
df

import matplotlib.pyplot as plt

offsets = {
    "OtherMamba": (-0.05, 0),
    "Point\nTransformer": (0.05, -0.08),
}

def plot_models(samples: pd.DataFrame, switch_view=True):

    marker_pool = ['o', 's', '^', 'D', 'P', '*', 'X']
    color_pool = plt.cm.tab10.colors

    tasks = sorted(samples["domain"].unique())
    archs = sorted(samples["type"].unique())

    if switch_view:
        color_key = {a: color_pool[i % len(color_pool)] for i, a in enumerate(archs)}
        marker_key = {t: marker_pool[i % len(marker_pool)] for i, t in enumerate(tasks)}
    else:
        color_key = {t: color_pool[i % len(color_pool)] for i, t in enumerate(tasks)}
        marker_key = {a: marker_pool[i % len(marker_pool)] for i, a in enumerate(archs)}

    fig, ax = plt.subplots(figsize=(7, 6))
    for idx, row in samples.iterrows():
        color = color_key[row["type"]] if switch_view else color_key[row["domain"]]
        marker = marker_key[row["domain"]] if switch_view else marker_key[row["type"]]

        ax.scatter(row["param_quantile"], row["score_quantile"], color=color, marker=marker, s=80)
        offset = offsets.get(row["name"], (0, 0))
        ax.annotate(row["name"], (row["param_quantile"] + offset[0], row["score_quantile"] + offset[1] + 0.02), textcoords="offset points", xytext=(0, 3), ha='center', fontsize=10)

    ax.set_xlabel("Parameters quantile", fontsize=13)
    ax.set_ylabel("Accuracy quantile", fontsize=13)


    xlims = ax.get_xlim()
    xs = xlims[1] - xlims[0]
    o = 0.1
    ax.set_xlim(xlims[0] - o * xs , xlims[1] + o * xs)


    ylims = ax.get_ylim()
    ys = ylims[1] - ylims[0]
    ax.set_ylim(ylims[0] - o * ys , ylims[1] + o * ys)

    color_labels = archs if switch_view else tasks
    marker_labels = tasks if switch_view else archs

    color_legend = [plt.Line2D([0], [0], marker='D', color='w',
                               label=label, markerfacecolor=color_key[label], markersize=10)
                    for label in color_labels]
    marker_legend = [plt.Line2D([0], [0], marker=marker_key[label], color='k',
                                label=label, linestyle='None', markersize=10)
                     for label in marker_labels]

    ax.legend(handles=color_legend + marker_legend)

    fig.savefig("./domaincomp.pdf")
    plt.show()


plot_models(df[df["domain"] != "language"], switch_view=True)
