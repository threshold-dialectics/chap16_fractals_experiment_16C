# summarize_results.py
import os
import json
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau


def load_latest_results_dir(base_dir="."):
    dirs = [d for d in os.listdir(base_dir) if d.startswith("results_") and os.path.isdir(d)]
    if not dirs:
        raise FileNotFoundError("No results directory found")
    latest = sorted(dirs)[-1]
    return latest


def compute_correlation_and_pvalues(df, essential_cols=None):
    df = df.copy()
    df["scenario_type_code"] = pd.factorize(df.get("scenario_type", []))[0]
    corr_cols = [
        "hurst_g_dot",
        "hurst_beta_dot",
        "hurst_f_crit_dot",
        "box_dim_2d",
        "avg_speed_pre_collapse",
        "avg_couple_pre_collapse",
        "speed_index_mean",
        "couple_index_mean",
        "speed_index_std",
        "couple_index_std",
        "time_to_failure_from_onset",
        "collapse_time",
        "scenario_type_code",
        "w1",
        "w2",
        "w3",
        "phi1",
    ] + [c for c in df.columns if c.startswith("param_")]
    corr_cols = [c for c in corr_cols if c in df.columns]
    essential_cols = [c for c in (essential_cols or ["hurst_beta_dot", "box_dim_2d"]) if c in corr_cols]
    corr_df = df[corr_cols]
    corr_df = corr_df.dropna(subset=essential_cols, how="any") if essential_cols else corr_df.dropna(how="all")
    valid_counts = corr_df.notna().sum()
    corr_df = corr_df[valid_counts[valid_counts >= 20].index]
    essential_cols = [c for c in essential_cols if c in corr_df.columns]
    if len(corr_df) > 1 and len(corr_df.columns) >= 2:
        corr_matrix = corr_df.corr()
        spearman_matrix = corr_df.corr(method="spearman")
        kendall_matrix = corr_df.corr(method="kendall")
        pval_matrix = pd.DataFrame(np.ones_like(corr_matrix), index=corr_matrix.index, columns=corr_matrix.columns)
        highlight = pd.DataFrame(False, index=corr_matrix.index, columns=corr_matrix.columns)
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i >= j:
                    continue
                valid = corr_df[[col1, col2]].dropna()
                if len(valid) > 1:
                    _, p = pearsonr(valid[col1], valid[col2])
                else:
                    p = np.nan
                pval_matrix.loc[col1, col2] = p
                pval_matrix.loc[col2, col1] = p
                highlight.loc[col1, col2] = highlight.loc[col2, col1] = bool(abs(corr_matrix.loc[col1, col2]) >= 0.3 and p < 0.01)
        return corr_matrix, pval_matrix, spearman_matrix, kendall_matrix, highlight
    else:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def grouped_stats(df):
    metrics = [
        "box_dim_2d",
        "hurst_beta_dot",
        "collapse_time",
        "box_dim_2d_precrisis",
        "hurst_beta_dot_precrisis",
    ]
    available = [m for m in metrics if m in df.columns]
    if not available:
        return {}, {}
    group_name = df.groupby("scenario_name")[available].agg(["mean", "median", "std"])
    group_type = df.groupby("scenario_type")[available].agg(["mean", "median", "std"])
    group_name.columns = ["{}_{}".format(m, stat) for m, stat in group_name.columns]
    group_type.columns = ["{}_{}".format(m, stat) for m, stat in group_type.columns]
    return group_name.to_dict(orient="index"), group_type.to_dict(orient="index")


def build_summary(results_dir):
    csv_path = os.path.join(results_dir, "td_fractal_study_1_results.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(results_dir, "td_fractal_experiment_1_results.csv")
    df = pd.read_csv(csv_path)
    collapse_rates = (df.groupby("scenario_name")["is_collapsed"].mean() * 100).to_dict()
    valid_percentages = {c: float(100 * df[c].notna().mean()) for c in ["hurst_g_dot", "hurst_beta_dot", "hurst_f_crit_dot", "box_dim_2d"] if c in df.columns}
    corr_matrix, pval_matrix, spear, kendall, highlight = compute_correlation_and_pvalues(df)
    group_name, group_type = grouped_stats(df)
    summary = {
        "collapse_rates": collapse_rates,
        "valid_data_percentages": valid_percentages,
        "correlation_matrix": corr_matrix.to_dict(),
        "p_value_matrix": pval_matrix.to_dict(),
        "spearman_matrix": spear.to_dict(),
        "kendall_matrix": kendall.to_dict(),
        "correlation_highlight": highlight.astype(bool).to_dict(),
        "grouped_stats_by_scenario_name": group_name,
        "grouped_stats_by_scenario_type": group_type,
    }
    return summary


def main():
    results_dir = load_latest_results_dir()
    summary = build_summary(results_dir)
    out_path = os.path.join(results_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {out_path}")


if __name__ == "__main__":
    main()
