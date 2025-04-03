import pandas as pd
max_lp = 4


samples = [
    ("video", "mamba",  "VideoMamba", 22.8, 64.2),
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
    ("point", "transformer", "PointGPT-L", 360, 94.9),
    ("point", "transformer", "PointGPT-S", 29.2, 94.0),
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

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

offsets = {
    "VideoMAE": (-0.02, 0),
    #"Point\nTransformer": (0.05, -0.08),
}

def plot_models(samples: pd.DataFrame, switch_view=True):
    samples = samples[samples["domain"] != "language"]
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

    image_mapping = {
        "CNN": "camera_blue.png",
        "mamba": "camera_orange.png",
        "transformer": "camera_green.png"
    }

    for idx, row in samples.iterrows():
        if row["domain"] == "video":
            img_file = image_mapping.get(row["type"])
            if img_file:
                img = plt.imread(img_file)
                oi = OffsetImage(img, zoom=0.05)
                ab = AnnotationBbox(oi, (row["param_quantile"], row["score_quantile"]), frameon=False)
                ax.add_artist(ab)
        else:
            color = color_key[row["type"]] if switch_view else color_key[row["domain"]]
            marker = marker_key[row["domain"]] if switch_view else marker_key[row["type"]]
            ax.scatter(row["param_quantile"], row["score_quantile"], color=color, marker=marker, s=80)
        offset = offsets.get(row["name"], (0, 0))
        ax.annotate(row["name"], (row["param_quantile"] + offset[0], row["score_quantile"] + offset[1] + 0.02),
                    textcoords="offset points", xytext=(0, 3), ha='center', fontsize=10)

    ax.set_xlabel("Parameters quantile", fontsize=13)
    ax.set_ylabel("Accuracy quantile", fontsize=13)

    xlims = ax.get_xlim()
    xs = xlims[1] - xlims[0]
    o = 0.1
    ax.set_xlim(xlims[0] - o * xs, xlims[1] + o * xs)

    ylims = ax.get_ylim()
    ys = ylims[1] - ylims[0]
    ax.set_ylim(ylims[0] - o * ys, ylims[1] + o * ys)

    color_labels = archs if switch_view else tasks
    marker_labels = tasks if switch_view else archs
    color_legend = [plt.Line2D([0], [0], marker='s', color='w',
                               label=label, markerfacecolor=color_key[label], markersize=10)
                    for label in color_labels]
    # marker_legend = [plt.Line2D([0], [0], marker=marker_key[label], color='k',
    #                             label=label, linestyle='None', markersize=10)
    #                  for label in marker_labels]
    ax.legend(handles=color_legend)

    fig.savefig("./domaincomp.pdf")
    plt.show()

plot_models(df, switch_view=True)


def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(
        int(rgb[0] * 255),
        int(rgb[1] * 255),
        int(rgb[2] * 255)
    )

[rgb_to_hex(c) for c in plt.cm.tab10.colors[:3]]