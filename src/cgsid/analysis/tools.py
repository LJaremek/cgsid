from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, TypedDict

import numpy as np
import pandas as pd

try:
    from IPython.display import display
except ImportError:  # pragma: no cover
    display = print


RelationSpec = tuple[str, str, str]
LabelMapping = Mapping[str, str]


class DirectedJSResult(TypedDict):
    weighted_js: float
    max_js: float
    strongest_condition_value: object | None
    detail: pd.DataFrame


COLUMN_LABELS: dict[str, str] = {
    "object_1": "Animal",
    "object_1_color": "Color",
    "object_2": "Second object",
    "object_2_color": "Second object color",
    "interaction": "Interaction",
    "pose": "Pose",
    "environment": "Environment",
    "day_time": "Time of day",
    "season": "Season",
}

VALUE_LABELS: dict[str, str] = {
    "holding_worm_in_beak": "worm in beak",
    "none": "none",
    "sitting_in_nest": "sitting in nest",
    "standing_on_branch": "standing on branch",
    "living_room": "living room",
    "day": "day",
    "evening": "evening",
}


def entropy(probabilities: Iterable[float]) -> float:
    values = np.asarray([p for p in probabilities if p > 0], dtype=float)
    if len(values) == 0:
        return 0.0
    return float(-(values * np.log2(values)).sum())


def js_divergence(p: Mapping[object, float], q: Mapping[object, float]) -> float:
    keys = sorted(set(p) | set(q), key=str)
    pv = np.asarray([p.get(k, 0.0) for k in keys], dtype=float)
    qv = np.asarray([q.get(k, 0.0) for k in keys], dtype=float)
    if pv.sum() > 0:
        pv = pv / pv.sum()
    if qv.sum() > 0:
        qv = qv / qv.sum()
    mv = 0.5 * (pv + qv)
    return entropy(mv) - 0.5 * entropy(pv) - 0.5 * entropy(qv)


def distribution(series: pd.Series) -> dict[object, float]:
    return series.value_counts(normalize=True, dropna=False).to_dict()


def directed_js(df: pd.DataFrame, condition_column: str, target_column: str) -> DirectedJSResult:
    base = distribution(df[target_column])
    rows: list[dict[str, object]] = []
    n = len(df)
    for value, sub in df.groupby(condition_column, dropna=False):
        score = js_divergence(distribution(sub[target_column]), base)
        weight = len(sub) / n if n else 0.0
        rows.append(
            {
                "condition_value": value,
                "count": len(sub),
                "weight": weight,
                "js": score,
                "weighted_js_component": score * weight,
            }
        )
    detail = pd.DataFrame(rows).sort_values("js", ascending=False)
    return {
        "weighted_js": float(detail["weighted_js_component"].sum()) if len(detail) else 0.0,
        "max_js": float(detail["js"].max()) if len(detail) else 0.0,
        "strongest_condition_value": detail.iloc[0]["condition_value"] if len(detail) else None,
        "detail": detail,
    }


def expected_relation_table(df: pd.DataFrame, relations: Sequence[RelationSpec]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, condition, target in relations:
        score = directed_js(df, condition, target)
        reverse = directed_js(df, target, condition)
        rows.append(
            {
                "relation": label,
                "condition_column": condition,
                "target_column": target,
                "weighted_js": score["weighted_js"],
                "max_js": score["max_js"],
                "strongest_condition_value": score["strongest_condition_value"],
                "reverse_weighted_js": reverse["weighted_js"],
            }
        )
    return pd.DataFrame(rows).sort_values("weighted_js", ascending=False).reset_index(drop=True)


def all_directed_js(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for condition in columns:
        for target in columns:
            if condition == target:
                continue
            score = directed_js(df, condition, target)
            rows.append(
                {
                    "relation": f"{condition} -> {target}",
                    "condition_column": condition,
                    "target_column": target,
                    "weighted_js": score["weighted_js"],
                    "max_js": score["max_js"],
                    "strongest_condition_value": score["strongest_condition_value"],
                }
            )
    return pd.DataFrame(rows).sort_values("weighted_js", ascending=False).reset_index(drop=True)


def crosstab_relation(df: pd.DataFrame, condition: str, target: str) -> pd.DataFrame:
    return pd.crosstab(df[condition], df[target], normalize="index").round(3)


def pretty_label(value: object) -> str:
    return VALUE_LABELS.get(str(value), str(value).replace("_", " "))


def value_distribution_table(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in frame.columns:
        counts = frame[column].value_counts(dropna=False).sort_values()
        proportions = frame[column].value_counts(normalize=True, dropna=False).reindex(counts.index)
        for value, count in counts.items():
            rows.append(
                {
                    "dataset": dataset,
                    "column": column,
                    "column_label": COLUMN_LABELS.get(column, column.replace("_", " ")),
                    "value": value,
                    "value_label": pretty_label(value),
                    "count": int(count),
                    "proportion": float(proportions.loc[value]),
                }
            )
    return pd.DataFrame(rows)


def plot_value_distributions(
    frame: pd.DataFrame,
    title: str,
    color: str = "#4f83b6",
    title_suffixes: Mapping[str, str] | None = None,
    value_suffixes: Mapping[tuple[str, object], str] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    frame = frame[[column for column in frame.columns if column in frame]]
    if frame.empty or len(frame.columns) == 0:
        print("No columns to plot")
        return

    n_cols = 2
    n_rows = int(np.ceil(len(frame.columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10.4, 10.4), sharex=True)
    axes = np.array(axes).reshape(-1)

    for ax, column in zip(axes, frame.columns):
        proportions = frame[column].value_counts(normalize=True, dropna=False).sort_values()
        labels = [pretty_label(value) for value in proportions.index]
        bars = ax.barh(labels, proportions.values, color=color, edgecolor="#263642", linewidth=0.7, alpha=0.92)

        column_title = COLUMN_LABELS.get(column, column.replace("_", " "))
        if title_suffixes and column in title_suffixes:
            column_title = f"{column_title} ({title_suffixes[column]})"
        ax.set_title(column_title, fontsize=10.5, fontweight="bold", pad=6)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Proportion of images", fontsize=9)
        ax.tick_params(axis="both", labelsize=8.5)
        ax.grid(axis="x", color="#d7dde3", linewidth=0.7, alpha=0.85)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#9aa6b2")

        for bar, value, label_value in zip(bars, proportions.values, proportions.index):
            value_text = f"{value:.2f}"
            if value_suffixes and (column, label_value) in value_suffixes:
                value_text = f"{value_text} | {value_suffixes[(column, label_value)]}"
            ax.text(
                min(value + 0.015, 0.98),
                bar.get_y() + bar.get_height() / 2,
                value_text,
                va="center",
                ha="left" if value < 0.92 else "right",
                fontsize=8.2,
                color="#1d2733",
            )

    for ax in axes[len(frame.columns):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    plt.show()


def argmax_label(df: pd.DataFrame, mapping: LabelMapping, default: object | None = None) -> pd.Series:
    available = {value: column for value, column in mapping.items() if column in df.columns}
    if not available:
        return pd.Series([default] * len(df), index=df.index)
    columns = [available[value] for value in available]
    winner = df[columns].idxmax(axis=1)
    reverse = {column: value for value, column in available.items()}
    result = winner.map(reverse)
    if default is not None:
        result = result.fillna(default)
    return result


def build_clip_metadata_from_mappings(
    clip: pd.DataFrame,
    mappings: Mapping[str, LabelMapping],
    defaults: Mapping[str, object | None] | None = None,
) -> pd.DataFrame:
    pred = pd.DataFrame(index=clip.index)
    defaults = defaults or {}
    for column, mapping in mappings.items():
        pred[column] = argmax_label(clip, mapping, default=defaults.get(column))
    return pred


def print_relation_tabs(df: pd.DataFrame, relations: Sequence[RelationSpec]) -> None:
    for label, condition, target in relations:
        print(f"\n=== {label}: {condition} -> {target} ===")
        display(crosstab_relation(df, condition, target))


def clip_accuracy_by_label(truth: pd.DataFrame, pred: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    truth = truth[list(columns)].reset_index(drop=True)
    pred = pred[list(columns)].reset_index(drop=True)

    for column in columns:
        correct = truth[column].eq(pred[column])
        rows.append(
            {
                "category": column,
                "category_label": COLUMN_LABELS.get(column, column.replace("_", " ")),
                "label": "__overall__",
                "label_display": "All labels",
                "correct": int(correct.sum()),
                "total": int(correct.size),
                "accuracy": float(correct.mean()) if len(correct) else np.nan,
            }
        )
        for label in sorted(truth[column].dropna().unique(), key=str):
            mask = truth[column].eq(label)
            label_correct = correct[mask]
            rows.append(
                {
                    "category": column,
                    "category_label": COLUMN_LABELS.get(column, column.replace("_", " ")),
                    "label": label,
                    "label_display": pretty_label(label),
                    "correct": int(label_correct.sum()),
                    "total": int(mask.sum()),
                    "accuracy": float(label_correct.mean()) if mask.any() else np.nan,
                }
            )

    result = pd.DataFrame(rows)
    result["accuracy_pct"] = 100 * result["accuracy"]
    return result


def format_clip_accuracy_table(accuracy: pd.DataFrame, *, dataset: str) -> Any:
    table = accuracy.copy()
    table["Category"] = table["category_label"]
    table["Label"] = table["label_display"]
    table["Correct"] = table["correct"]
    table["Total"] = table["total"]
    table["Accuracy [%]"] = table["accuracy_pct"]
    table["Share of dataset [%]"] = 100 * table["total"] / table["total"].max()
    table = table[["Category", "Label", "Correct", "Total", "Accuracy [%]", "Share of dataset [%]"]]

    category_order = {category: i for i, category in enumerate(accuracy["category_label"].drop_duplicates())}
    table["_category_order"] = table["Category"].map(category_order)
    table["_label_order"] = table["Label"].eq("All labels").map({True: 0, False: 1})
    table = table.sort_values(["_category_order", "_label_order", "Accuracy [%]"], ascending=[True, True, False])
    table = table.drop(columns=["_category_order", "_label_order"])

    styles = [
        {"selector": "caption", "props": "caption-side: top; text-align: left; font-size: 15px; font-weight: 700; margin-bottom: 8px;"},
        {"selector": "th", "props": "background-color: #eef2f6; color: #1f2933; font-weight: 700; border-bottom: 1px solid #c7d0d9;"},
        {"selector": "td", "props": "border-bottom: 1px solid #e4e8ee;"},
    ]

    return (
        table.style
        .hide(axis="index")
        .set_caption(f"CLIP accuracy by category and label in the {dataset} dataset")
        .format({"Accuracy [%]": "{:.2f}", "Share of dataset [%]": "{:.2f}"})
        .bar(subset=["Accuracy [%]"], vmin=0, vmax=100, color="#9fc5e8")
        .background_gradient(subset=["Accuracy [%]"], vmin=0, vmax=100, cmap="RdYlGn")
        .set_table_styles(styles)
        .set_properties(subset=["Category", "Label"], **{"text-align": "left"})
        .set_properties(subset=["Correct", "Total", "Accuracy [%]", "Share of dataset [%]"], **{"text-align": "right"})
    )
