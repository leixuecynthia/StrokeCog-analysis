#!/usr/bin/env python3
"""
Code availability: Machine-learning classifier + correlation-network analysis.

Input format (CSV):
  - One row per sample.
  - A binary label column (default: "label", values 0/1).
  - All other columns are numeric features.
  - Optional ID column (default: "SampleId") is ignored.

Outputs (in --outdir):
  - cv_metrics.csv, cv_summary.json
  - model_coefficients.csv
  - correlation_network.gml, correlation_network_mst.gml
  - network_summary.json
  - correlation_network.png, correlation_network_mst.png
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import networkx as nx

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.manifold import TSNE

try:
    import community as community_louvain  # type: ignore
except Exception:  # pragma: no cover
    import community_louvain  # type: ignore


# -----------------------------
# Data I/O
# -----------------------------

def load_features_table(
    csv_path: str | Path,
    label_col: str = "label",
    id_col: Optional[str] = "SampleId",
) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        raise ValueError(f"label_col='{label_col}' not found. Columns: {list(df.columns)[:10]}...")

    y = df[label_col].astype(int)
    unique = sorted(pd.unique(y))
    if not set(unique).issubset({0, 1}) or len(unique) < 2:
        raise ValueError(f"Label column must be binary (0/1). Found: {unique}")

    drop_cols = [label_col]
    if id_col and id_col in df.columns:
        drop_cols.append(id_col)

    X = df.drop(columns=drop_cols)
    X = X.select_dtypes(include=[np.number]).copy()
    if X.shape[1] == 0:
        raise ValueError("No numeric feature columns found after dropping label/id columns.")

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    return X, y


def read_feature_list(path: str | Path) -> List[str]:
    features: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                features.append(name)
    return features


# -----------------------------
# ML model (Elastic-net logistic regression)
# -----------------------------

@dataclass
class CVResult:
    auroc_per_fold: List[float]
    auprc_per_fold: List[float]
    mean_auroc: float
    mean_auprc: float


def build_elasticnet_logreg(l1_ratio: float = 0.5, random_state: int = 66) -> Pipeline:
    model = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=l1_ratio,
        max_iter=10000,
        random_state=random_state,
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", model)])


def repeated_5fold_cv_eval(
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: int = 5,
    repeats: int = 1,
    l1_ratio: float = 0.5,
    random_state: int = 66,
) -> CVResult:
    """
    5-fold stratified CV (optionally repeated). Set repeats=1 to run standard 5-fold CV.
    """
    rkf = RepeatedStratifiedKFold(
        n_splits=cv_splits,
        n_repeats=repeats,
        random_state=random_state,
    )
    pipe = build_elasticnet_logreg(l1_ratio=l1_ratio, random_state=random_state)

    aurocs: List[float] = []
    auprcs: List[float] = []

    for train_idx, test_idx in rkf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipe.fit(X_train, y_train)
        y_score = pipe.predict_proba(X_test)[:, 1]

        aurocs.append(float(roc_auc_score(y_test, y_score)))
        auprcs.append(float(average_precision_score(y_test, y_score)))

    return CVResult(
        auroc_per_fold=aurocs,
        auprc_per_fold=auprcs,
        mean_auroc=float(np.mean(aurocs)),
        mean_auprc=float(np.mean(auprcs)),
    )


def fit_full_model(X: pd.DataFrame, y: pd.Series, l1_ratio: float = 0.5, random_state: int = 66) -> Pipeline:
    pipe = build_elasticnet_logreg(l1_ratio=l1_ratio, random_state=random_state)
    pipe.fit(X, y)
    return pipe


# -----------------------------
# Correlation network
# -----------------------------

def pick_topk_features_by_label_corr(X: pd.DataFrame, y: pd.Series, topk: int = 500) -> List[str]:
    y_centered = (y - y.mean()).to_numpy()
    scores: Dict[str, float] = {}
    for col in X.columns:
        x = X[col].to_numpy()
        x = x - np.nanmean(x)
        denom = (np.sqrt(np.nanvar(x)) * np.sqrt(np.nanvar(y_centered)))
        if denom == 0 or np.isnan(denom):
            scores[col] = 0.0
        else:
            scores[col] = float(np.nanmean(x * y_centered) / denom)

    ranked = sorted(scores.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return [k for k, _ in ranked[: min(topk, len(ranked))]]


def build_correlation_graph(
    X: pd.DataFrame,
    features: List[str],
    method: str = "spearman",
    threshold: float = 0.5,
) -> Tuple[nx.Graph, pd.DataFrame]:
    sub = X[features].copy()
    corr = sub.corr(method=method)

    edges = []
    cols = corr.columns.to_list()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iat[i, j]
            if np.isfinite(r) and abs(r) >= threshold:
                edges.append((cols[i], cols[j], float(r)))

    G = nx.Graph()
    G.add_nodes_from(features)
    for u, v, r in edges:
        G.add_edge(u, v, correlation=r)

    # Remove isolates to keep plots readable
    G.remove_nodes_from(list(nx.isolates(G)))
    return G, corr


def louvain_communities(G: nx.Graph, random_state: int = 66) -> Dict[str, int]:
    if G.number_of_edges() == 0:
        return {n: 0 for n in G.nodes()}
    return community_louvain.best_partition(G, random_state=random_state)


def minimum_spanning_tree_by_abs_corr(G: nx.Graph) -> nx.Graph:
    H = nx.Graph()
    for u, v, d in G.edges(data=True):
        r = float(d.get("correlation", 0.0))
        H.add_edge(u, v, weight=1.0 - abs(r), correlation=r)
    if H.number_of_edges() == 0:
        return H
    return nx.minimum_spanning_tree(H, weight="weight")


def tsne_layout_from_corr(
    G: nx.Graph,
    corr_matrix: pd.DataFrame,
    seed: int = 66,
) -> Dict[str, np.ndarray]:
    """
    Compute a 2D t-SNE embedding from the feature-feature correlation matrix,
    then use those coordinates to plot the network.
    """
    nodes = list(G.nodes())
    if len(nodes) == 0:
        return {}
    if len(nodes) == 1:
        return {nodes[0]: np.array([0.0, 0.0])}

    cm = corr_matrix.loc[nodes, nodes].to_numpy()

    perplexity = min(30, max(2, len(nodes) - 1))
    tsne = TSNE(
        n_components=2,
        random_state=seed,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
    )
    emb = tsne.fit_transform(cm)
    return {nodes[i]: emb[i] for i in range(len(nodes))}


def plot_network(
    G: nx.Graph,
    partition: Dict[str, int],
    pos: Dict[str, np.ndarray],
    out_png: str | Path,
    title: str,
    node_size: int = 40,
    with_labels: bool = False,
) -> None:
    out_png = str(out_png)
    if G.number_of_nodes() == 0:
        raise ValueError("Graph has 0 nodes. Try lowering threshold or increasing top-k features.")

    communities = np.array([partition.get(n, 0) for n in G.nodes()])
    plt.figure(figsize=(12, 10))
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.7)
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=communities)
    if with_labels:
        nx.draw_networkx_labels(G, pos, font_size=7)

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def save_network_summary(G: nx.Graph, partition: Dict[str, int], out_json: str | Path) -> None:
    out_json = str(out_json)
    comm_sizes: Dict[int, int] = {}
    for n in G.nodes():
        cid = int(partition.get(n, 0))
        comm_sizes[cid] = comm_sizes.get(cid, 0) + 1

    summary = {
        "n_nodes": int(G.number_of_nodes()),
        "n_edges": int(G.number_of_edges()),
        "density": float(nx.density(G)) if G.number_of_nodes() > 1 else 0.0,
        "n_connected_components": int(nx.number_connected_components(G)) if G.number_of_nodes() else 0,
        "community_sizes": {str(k): int(v) for k, v in sorted(comm_sizes.items())},
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="CSV with label + numeric feature columns")
    p.add_argument("--label-col", default="label", help="Binary label column name (0/1)")
    p.add_argument("--id-col", default="SampleId", help="Optional ID column to ignore if present")
    p.add_argument("--outdir", default="results", help="Output directory")

    # ML (5-fold CV)
    p.add_argument("--cv-splits", type=int, default=5)
    p.add_argument("--repeats", type=int, default=1, help="Repeat 5-fold CV N times (default 1)")
    p.add_argument("--l1-ratio", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=66)

    # Network
    p.add_argument("--features-list", default=None, help="Optional text file of feature names to use for network")
    p.add_argument("--net-topk", type=int, default=500, help="If no feature list is given, select top-k by label corr")
    p.add_argument("--net-method", default="spearman", choices=["spearman", "pearson"])
    p.add_argument("--net-threshold", type=float, default=0.5, help="Keep edges with |corr| >= threshold")
    p.add_argument("--net-layout", default="tsne", choices=["tsne"], help="Network plot layout (tsne)")
    p.add_argument("--net-labels", action="store_true", help="Draw node labels (can be cluttered)")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    X, y = load_features_table(args.data, label_col=args.label_col, id_col=args.id_col)

    # ML (5-fold CV)
    cvres = repeated_5fold_cv_eval(
        X, y,
        cv_splits=args.cv_splits,
        repeats=args.repeats,
        l1_ratio=args.l1_ratio,
        random_state=args.seed,
    )

    metrics_df = pd.DataFrame({
        "fold": np.arange(len(cvres.auroc_per_fold)),
        "auroc": cvres.auroc_per_fold,
        "auprc": cvres.auprc_per_fold,
    })
    metrics_df.to_csv(outdir / "cv_metrics.csv", index=False)

    summary = {
        "mean_auroc": cvres.mean_auroc,
        "mean_auprc": cvres.mean_auprc,
        "cv_splits": int(args.cv_splits),
        "repeats": int(args.repeats),
        "l1_ratio": float(args.l1_ratio),
        "seed": int(args.seed),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
    }
    with open(outdir / "cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Fit full model and save coefficients (interpretable)
    pipe = fit_full_model(X, y, l1_ratio=args.l1_ratio, random_state=args.seed)
    clf: LogisticRegression = pipe.named_steps["clf"]
    coefs = pd.Series(clf.coef_.ravel(), index=X.columns, name="coef").sort_values(key=np.abs, ascending=False)
    coefs.to_csv(outdir / "model_coefficients.csv")

    # Correlation network
    if args.features_list:
        features = [f for f in read_feature_list(args.features_list) if f in X.columns]
        if len(features) == 0:
            raise ValueError("None of the provided features were found in the dataset columns.")
    else:
        features = pick_topk_features_by_label_corr(X, y, topk=args.net_topk)

    G, corr_mat = build_correlation_graph(
        X=X,
        features=features,
        method=args.net_method,
        threshold=args.net_threshold,
    )
    part = louvain_communities(G, random_state=args.seed)
    mst = minimum_spanning_tree_by_abs_corr(G)
    part_mst = {n: part.get(n, 0) for n in mst.nodes()}

    pos = tsne_layout_from_corr(G, corr_mat, seed=args.seed)
    # For MST, reuse the same node coordinates (keeps visual consistency)
    pos_mst = {n: pos.get(n, np.array([0.0, 0.0])) for n in mst.nodes()}

    # Save network outputs
    nx.write_gml(G, outdir / "correlation_network.gml")
    nx.write_gml(mst, outdir / "correlation_network_mst.gml")
    save_network_summary(G, part, outdir / "network_summary.json")

    # Plots
    plot_network(
        G, part, pos,
        out_png=outdir / "correlation_network.png",
        title=f"Correlation network (|r| >= {args.net_threshold}, {args.net_method}, layout=tSNE)",
        with_labels=args.net_labels,
    )
    plot_network(
        mst, part_mst, pos_mst,
        out_png=outdir / "correlation_network_mst.png",
        title="Minimum spanning tree (distance = 1 - |corr|, layout=tSNE)",
        with_labels=args.net_labels,
    )

    print("=== ML (5-fold stratified CV) ===")
    print(f"Mean AUROC: {cvres.mean_auroc:.4f}")
    print(f"Mean AUPRC: {cvres.mean_auprc:.4f}")
    print(f"Saved: {outdir/'cv_metrics.csv'} and {outdir/'cv_summary.json'}")

    print("\n=== Network ===")
    print(f"Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}")
    print("Layout: t-SNE on correlation matrix")
    print(f"Saved: {outdir/'correlation_network.png'} and {outdir/'correlation_network_mst.png'}")


if __name__ == "__main__":
    main()
