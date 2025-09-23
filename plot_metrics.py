#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8")

# ---------- CONFIG ----------
METRIC_KEY = {
    "fiedler": "fiedler_value",
    "lambda2": "fiedler_value",
    "λ2": "fiedler_value",
    "hfer": "hfer",
    "energy": "energy",
    "smoothness": "smoothness_index",
    "smi": "smoothness_index",
    "spectral_entropy": "spectral_entropy",
    "entropy": "spectral_entropy",
}
METRIC_LABEL = {
    "fiedler": r"$\lambda_2$",
    "lambda2": r"$\lambda_2$",
    "λ2": r"$\lambda_2$",
    "hfer": "HFER",
    "energy": "Energy",
    "smoothness": "Smoothness Index",
    "smi": "Smoothness Index",
    "spectral_entropy": "Spectral Entropy",
    "entropy": "Spectral Entropy",
}
ALL_CANONICAL = ["fiedler", "hfer", "energy", "smoothness", "spectral_entropy"]

# ---------- IO HELPERS ----------
def _pick_json(pathlike: str) -> str | None:
    p = Path(pathlike)
    candidates = []
    if any(ch in pathlike for ch in "*?[]"):
        candidates = [Path(x) for x in glob(pathlike)]
    elif p.is_dir():
        candidates = sorted(p.glob("diagnostics_*.json")) or sorted(p.glob("*.json"))
    elif p.is_file():
        candidates = [p]
    else:
        return None
    if not candidates:
        return None
    diags = [c for c in candidates if c.name.startswith("diagnostics_")]
    if diags:
        return str(sorted(diags, key=lambda x: x.stat().st_mtime)[-1])
    return str(sorted(candidates, key=lambda x: x.stat().st_mtime)[-1])

def _extract_layers_metric(json_path: str, metric_key: str) -> tuple[np.ndarray, np.ndarray]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "diagnostics" in data and isinstance(data["diagnostics"], list):
        L, V = [], []
        for d in data["diagnostics"]:
            if isinstance(d, dict) and (metric_key in d):
                L.append(int(d.get("layer", len(L))))
                V.append(float(d[metric_key]))
        return np.array(L), np.array(V, dtype=float)

    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, dict) and "layers" in v and isinstance(v["layers"], list):
                L = [int(x.get("layer", i)) for i, x in enumerate(v["layers"])]
                V = [float(x.get(metric_key, np.nan)) for x in v["layers"]]
                return np.array(L), np.array(V, dtype=float)

    if isinstance(data, list) and data and isinstance(data[0], dict) and metric_key in data[0]:
        L = [int(x.get("layer", i)) for i, x in enumerate(data)]
        V = [float(x.get(metric_key, np.nan)) for x in data]
        return np.array(L), np.array(V, dtype=float)

    raise ValueError(f"Unrecognized JSON or missing metric '{metric_key}': {json_path}")

def _align_delta(active_path: str, passive_path: str, metric_key: str) -> tuple[np.ndarray, np.ndarray]:
    apath = _pick_json(active_path)
    ppath = _pick_json(passive_path)
    if apath is None or ppath is None:
        raise FileNotFoundError(f"Missing JSON: active={active_path}, passive={passive_path}")
    la, va = _extract_layers_metric(apath, metric_key)
    lp, vp = _extract_layers_metric(ppath, metric_key)

    common = np.intersect1d(la, lp)
    if common.size == 0:
        raise ValueError(f"No overlapping layers between {apath} and {ppath}")
    idx_a = {int(l): i for i, l in enumerate(la)}
    idx_p = {int(l): i for i, l in enumerate(lp)}
    va_c = np.array([va[idx_a[int(l)]] for l in common], dtype=float)
    vp_c = np.array([vp[idx_p[int(l)]] for l in common], dtype=float)
    return common.astype(int), (vp_c - va_c)

def _align_single(path: str, metric_key: str) -> tuple[np.ndarray, np.ndarray]:
    j = _pick_json(path)
    if j is None:
        raise FileNotFoundError(f"Missing JSON: {path}")
    return _extract_layers_metric(j, metric_key)

# ---------- UTILS ----------
def make_group_colors(group_names: list[str]) -> dict[str, tuple]:
    fixed = {
        "Americas": "#1f77b4", "Europe": "#2ca02c", "Asia": "#9467bd",
        "Africa": "#d62728", "Oceania": "#ff7f0e",
        "analytic": "#1f77b4", "periphrastic": "#2ca02c", "affixal": "#d62728",
        "particle": "#9467bd", "non-concatenative": "#ff7f0e", "unknown": "#7f7f7f",
    }
    cmap = plt.get_cmap("tab20")
    colors = {}
    i = 0
    for g in group_names:
        if g in fixed:
            colors[g] = fixed[g]
        else:
            colors[g] = cmap(i % cmap.N); i += 1
    return colors

def place_legend(ax, n_items: int):
    if n_items <= 6:
        ax.legend(loc="upper right", fontsize=9, ncol=min(3, n_items))
    elif n_items <= 12:
        ax.legend(loc="upper right", fontsize=8, ncol=2)
    else:
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
                  fontsize=8, frameon=False, ncol=1)

def bootstrap_ci(x, n_boot: int = 2000, alpha: float = 0.05, seed: int = 1234):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan, (np.nan, np.nan)
    boots = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    lo = np.percentile(boots, 100 * (alpha / 2))
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    return float(x.mean()), (float(lo), float(hi))

# ---------- CORE (single-metric plot) ----------
def plot_from_csv(csv_path: str, metric_norm: str, delta: bool, early_lo: int, early_hi: int,
                  show_individual: bool, sort_mode: str, out_path: str | None):
    if metric_norm not in METRIC_KEY:
        raise ValueError(f"Unknown metric '{metric_norm}'. Options: {', '.join(METRIC_KEY.keys())}")
    metric_key = METRIC_KEY[metric_norm]
    metric_label = METRIC_LABEL.get(metric_norm, metric_norm)

    df = pd.read_csv(csv_path)

    per_group: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    counts: dict[str, int] = {}
    for _, row in df.iterrows():
        label = str(row["label"])
        group = str(row["continent"])
        active = str(row["active"])
        passive = str(row["passive"])
        try:
            if delta:
                layers, series = _align_delta(active, passive, metric_key)
            else:
                layers, series = _align_single(active, metric_key)
        except Exception as e:
            print(f"⚠️ Skipping {label} ({group}): {e}")
            continue
        per_group.setdefault(group, []).append((layers, series))
        counts[group] = counts.get(group, 0) + 1

    groups = sorted(per_group.keys())
    if not groups:
        raise SystemExit("No valid rows to plot.")
    group_colors = make_group_colors(groups)

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.0, 2.0], hspace=0.35, wspace=0.25)
    ax_main = fig.add_subplot(gs[0, :])
    ax_zoom = fig.add_subplot(gs[1, 0])
    ax_bar  = fig.add_subplot(gs[1, 1])

    # MAIN: mean ± std per group
    for grp in groups:
        series = per_group[grp]
        if not series:
            continue
        color = group_colors[grp]
        min_len = min(s[0].size for s in series)
        ref_layers = series[0][0][:min_len]
        aligned = np.vstack([s[1][:min_len] for s in series])

        mean = np.nanmean(aligned, axis=0)
        std  = np.nanstd(aligned, axis=0)

        ax_main.plot(ref_layers, mean, label=f"{grp} (n={counts[grp]})", color=color, linewidth=2.2)
        ax_main.fill_between(ref_layers, mean-std, mean+std, color=color, alpha=0.12, linewidth=0)

        if show_individual:
            for (L, v) in series:
                ax_main.plot(L[:min_len], v[:min_len], color=color, alpha=0.25, linewidth=0.9)

    if delta:
        ax_main.axhline(0.0, color="k", lw=1, alpha=0.6)

    ax_main.set_xlabel("Layer")
    ylab = (r"$\Delta$ " if delta else "") + metric_label
    if delta and metric_norm in {"fiedler", "lambda2", "λ2"}:
        ylab += r"  (passive − active)"
    ax_main.set_ylabel(ylab)

    title_metric = METRIC_LABEL.get(metric_norm, metric_label)
    ax_main.set_title(
        f"{title_metric} by group (mean ± std){' — Δ = passive − active' if delta else ''}"
    )
    place_legend(ax_main, len(groups))

    # ZOOM: early window layerwise mean
    early_means = {}
    for grp in groups:
        series = per_group[grp]
        min_len = min(s[0].size for s in series)
        ref_layers = series[0][0][:min_len]
        aligned = np.vstack([s[1][:min_len] for s in series])
        mask = (ref_layers >= early_lo) & (ref_layers <= early_hi)
        if not mask.any():
            continue
        layerwise_mean = np.nanmean(aligned, axis=0)
        ax_zoom.plot(ref_layers[mask], layerwise_mean[mask],
                     color=group_colors[grp], marker="o", lw=2, label=f"{grp} (n={counts[grp]})")
        early_means[grp] = np.nanmean(layerwise_mean[mask])

    if delta:
        ax_zoom.axhline(0.0, color="k", lw=1, alpha=0.6)
    ax_zoom.set_title(f"Zoom: layers {early_lo}–{early_hi}")
    ax_zoom.set_xlabel("Layer")
    ax_zoom.set_ylabel((r"$\Delta$ " if delta else "") + metric_label)
    place_legend(ax_zoom, len(early_means))

    # BAR: bootstrap CI of early-window mean across languages (per group)
    names, means, los, his = [], [], [], []
    for grp in groups:
        series = per_group[grp]
        lang_vals = []
        for L, v in series:
            mask = (L >= early_lo) & (L <= early_hi)
            if mask.any():
                lang_vals.append(np.nanmean(v[mask]))
        if not lang_vals:
            continue
        b_mean, (lo, hi) = bootstrap_ci(np.array(lang_vals))
        names.append(grp); means.append(b_mean); los.append(lo); his.append(hi)

    if names:
        names = np.array(names)
        means = np.array(means); los = np.array(los); his = np.array(his)

        if sort_mode == "name":
            order = np.argsort(names)
        elif sort_mode == "mean":
            order = np.argsort(means)[::-1]
        else:  # absmean
            order = np.argsort(np.abs(means))[::-1]

        names, means, los, his = names[order], means[order], los[order], his[order]
        bar_colors = [group_colors[n] for n in names]

        bars = ax_bar.bar(names, means, color=bar_colors, alpha=0.95)
        ax_bar.errorbar(names, means, yerr=[means - los, his - means],
                        fmt="none", ecolor="k", elinewidth=1.2, capsize=4, zorder=3)

        # annotate bars with mean values
        for rect, m in zip(bars, means):
            ax_bar.text(rect.get_x() + rect.get_width()/2, rect.get_height(),
                        f"{m:+.3f}", ha="center", va="bottom",
                        fontsize=8, rotation=0, color="#222")

        if len(names) > 8:
            ax_bar.set_xticklabels(names, rotation=35, ha="right")

    if delta:
        ax_bar.axhline(0.0, color="k", lw=1, alpha=0.6)
        ax_bar.set_title(
            f"Early-window mean Δ{title_metric}  (layers {early_lo}-{early_hi})\nbootstrap 95% CI"
        )
        ax_bar.set_ylabel(r"$\Delta$ " + title_metric)
    else:
        ax_bar.set_title(
            f"Early-window mean {title_metric}  (layers {early_lo}-{early_hi})\nbootstrap 95% CI"
        )
        ax_bar.set_ylabel(title_metric)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"✅ Saved: {out_path}")
    plt.show()

# ---------- PLOTTING ----------
def main():
    p = argparse.ArgumentParser(description="Metric by group (mean±std), with optional Δ=passive−active.")
    p.add_argument("--csv", required=True,
                   help="CSV columns: label,continent,active,passive (active/passive dirs or JSONs)")
    p.add_argument("--metric", required=True,
                   help="One of: fiedler|lambda2|λ2|hfer|energy|smoothness|smi|spectral_entropy|entropy|all")
    p.add_argument("--delta", action="store_true",
                   help="Plot Δ = passive − active (else absolute from 'active')")
    p.add_argument("--early_lo", type=int, default=2, help="Early window start layer (inclusive)")
    p.add_argument("--early_hi", type=int, default=5, help="Early window end layer (inclusive)")
    p.add_argument("--show_individual", action="store_true", help="Overlay each language series")
    p.add_argument("--sort", choices=["name", "mean", "absmean"], default="absmean",
                   help="Sort bars in the CI panel")
    p.add_argument("--out", default=None, help="Output PNG (base file). In --metric all, suffix _<metric> is added.")
    args = p.parse_args()

    metric_arg = args.metric.lower().strip()

    # MULTI-METRIC: loop over canonical set and write a separate file for each
    if metric_arg == "all":
        for m in ALL_CANONICAL:
            # derive output path per metric
            if args.out:
                stem = Path(args.out).with_suffix("")
                ext = Path(args.out).suffix or ".png"
                out_path = f"{stem}_{m}{ext}"
            else:
                out_path = f"metric_{m}.png"
            plot_from_csv(
                csv_path=args.csv,
                metric_norm=m,
                delta=args.delta,
                early_lo=args.early_lo,
                early_hi=args.early_hi,
                show_individual=args.show_individual,
                sort_mode=args.sort,
                out_path=out_path,
            )
        return

    # SINGLE METRIC
    # Normalize to a known key (keep synonyms)
    if metric_arg not in METRIC_KEY:
        raise ValueError(f"Unknown metric '{args.metric}'. Options: {', '.join(METRIC_KEY.keys())} or 'all'")
    out_path_single = args.out
    plot_from_csv(
        csv_path=args.csv,
        metric_norm=metric_arg,
        delta=args.delta,
        early_lo=args.early_lo,
        early_hi=args.early_hi,
        show_individual=args.show_individual,
        sort_mode=args.sort,
        out_path=out_path_single,
    )

if __name__ == "__main__":
    main()
