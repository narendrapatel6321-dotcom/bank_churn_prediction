import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_class_imbalance(train_df: pd.DataFrame) -> None:
    """
    Plot a side-by-side count bar and pie chart showing the churn class split.

    Parameters
    ----------
    train_df : Training DataFrame containing an 'Exited' column.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    palette   = sns.color_palette("muted")

    counts = train_df["Exited"].value_counts()
    axes[0].bar(["Retained", "Churned"], counts.values,
                color=[palette[0], palette[3]], alpha=0.8)
    axes[0].set_title("Customer Count by Churn Status")
    axes[0].set_ylabel("Count")

    pcts = train_df["Exited"].value_counts(normalize=True) * 100
    axes[1].pie(pcts, labels=["Retained", "Churned"],
                colors=[palette[0], palette[3]], autopct="%1.1f%%")
    axes[1].set_title("Churn Proportion")

    plt.suptitle("Class Imbalance Overview", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()

def plot_kde_by_churn(train_df: pd.DataFrame, cols: list[str]) -> None:
    """
    Plot KDE distributions of numeric columns split by churn status.

    Parameters
    ----------
    train_df : Training DataFrame with an 'Exited' column.
    cols     : List of numeric column names to plot (one subplot each).
    """
    fig, axes = plt.subplots(1, len(cols), figsize=(6 * len(cols), 5))
    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        sns.kdeplot(data=train_df, x=col, hue="Exited",
                    fill=True, common_norm=False, ax=ax)
        ax.set_title(f"{col} Distribution by Churn")
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, ["Retained (0)", "Churned (1)"])

    plt.tight_layout()
    plt.show()

def plot_churn_rate_bar(
    df: pd.DataFrame,
    group_col: str,
    x_labels: list[str] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
) -> None:
    """
    Plot churn rate (%) per category in group_col with an overall average line.

    Parameters
    ----------
    df        : DataFrame with group_col and 'Exited' columns.
    group_col : Column to group by (e.g. 'Geography', 'NumOfProducts').
    x_labels  : Human-readable tick labels. Defaults to the category values.
    title     : Plot title. Defaults to f'Churn Rate by {group_col}'.
    xlabel    : Optional x-axis label.
    """
    churn_rate = df.groupby(group_col)["Exited"].mean() * 100
    x_ticks    = x_labels if x_labels is not None else churn_rate.index.astype(str)
    palette    = sns.color_palette("muted")

    fig, ax = plt.subplots(figsize=(8, 4))
    bars    = ax.bar(x_ticks, churn_rate.values, color=palette[0], alpha=0.8)

    ax.axhline(df["Exited"].mean() * 100, color="grey",
               linestyle="--", lw=1.5, label="Overall average")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_title(title or f"Churn Rate by {group_col}")
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.legend()

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{bar.get_height():.1f}%",
            ha="center", fontsize=10,
        )

    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(train_df: pd.DataFrame, threshold: float = 0.7) -> None:
    """
    Plot a lower-triangle correlation heatmap and print any high-correlation pairs.

    Parameters
    ----------
    train_df  : Training DataFrame (numeric columns extracted automatically).
    threshold : Correlation magnitude above which pairs are flagged. Default 0.7.
    """
    numeric_cols = train_df.select_dtypes(include="number")
    corr_matrix = numeric_cols.corr()
    mask         = np.triu(np.ones_like(corr_matrix, dtype=bool))

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(numeric_cols.corr(), mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    corr_matrix = corr_matrix.abs()
    high_corr   = [
        (c1, c2, corr_matrix.loc[c1, c2])
        for c1 in corr_matrix.columns
        for c2 in corr_matrix.columns
        if c1 < c2 and corr_matrix.loc[c1, c2] > threshold
    ]

    if high_corr:
        print(f"High correlations (>{threshold}):")
        for c1, c2, val in high_corr:
            print(f"   {c1} <> {c2} : {val:.3f}")
    else:
        print("No redundant features found.")
