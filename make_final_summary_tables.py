import pandas as pd
from pathlib import Path
import numpy as np

OUT = Path("outputs_manuscript_150_ta099")
TABLE_DIR = OUT / "tables"
METRICS = TABLE_DIR / "metrics_raw.csv"

if not METRICS.exists():
    raise FileNotFoundError(f"Arquivo não encontrado: {METRICS}")

df = pd.read_csv(METRICS)

model_order = [
    "OLS_ANOVA_mass_fixed_effects",
    "Bayes_hierarchical_mass_MCMC",
    "PVS_aware_area_MCMC",
]

model_labels = {
    "OLS_ANOVA_mass_fixed_effects": "OLS/ANOVA mass-based",
    "Bayes_hierarchical_mass_MCMC": "Bayesian hierarchical mass-based",
    "PVS_aware_area_MCMC": "PVS-aware area-based",
}

def fmt_median_iqr(series, digits=4):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return "NA"
    q1, med, q3 = s.quantile([0.25, 0.50, 0.75])
    return f"{med:.{digits}f} [{q1:.{digits}f}, {q3:.{digits}f}]"

def divergence_summary(g):
    if "n_divergences" not in g.columns:
        return "NA"
    s = pd.to_numeric(g["n_divergences"], errors="coerce").dropna()
    if len(s) == 0:
        return "NA"
    n = len(s)
    n_with = int((s > 0).sum())
    total = int(s.sum())
    q1, med, q3 = s.quantile([0.25, 0.50, 0.75])
    maxv = int(s.max())
    return f"{n_with}/{n}; total={total}; median={med:.0f} [{q1:.0f}, {q3:.0f}]; max={maxv}"

def rhat_warnings(g, threshold=1.01):
    if "rhat_max" not in g.columns:
        return "NA"
    s = pd.to_numeric(g["rhat_max"], errors="coerce").dropna()
    if len(s) == 0:
        return "NA"
    return f"{int((s > threshold).sum())}/{len(s)}"

def ess_warnings(g, threshold=200):
    ess_cols = [c for c in ["ess_bulk_min", "ess_tail_min"] if c in g.columns]
    if not ess_cols:
        return "NA"

    sub = g[ess_cols].apply(pd.to_numeric, errors="coerce")
    sub = sub.dropna(how="all")

    if len(sub) == 0:
        return "NA"

    any_low = (sub < threshold).any(axis=1)
    bulk_low = (sub["ess_bulk_min"] < threshold).sum() if "ess_bulk_min" in sub.columns else 0
    tail_low = (sub["ess_tail_min"] < threshold).sum() if "ess_tail_min" in sub.columns else 0

    return f"{int(any_low.sum())}/{len(sub)}; bulk={int(bulk_low)}; tail={int(tail_low)}"

def make_summary(data, label):
    rows = []

    for model in model_order:
        g = data[data["model_name"] == model].copy()
        if len(g) == 0:
            continue

        rows.append({
            "model_name": model_labels.get(model, model),
            "n": len(g),
            "RMSE median [IQR]": fmt_median_iqr(g["rmse"]) if "rmse" in g.columns else "NA",
            "MAE median [IQR]": fmt_median_iqr(g["mae"]) if "mae" in g.columns else "NA",
            "PPC coverage median [IQR]": fmt_median_iqr(g["ppc_coverage"]) if "ppc_coverage" in g.columns else "NA",
            "PVS theta median [IQR]": fmt_median_iqr(g["pvs_theta"]) if "pvs_theta" in g.columns else "NA",
            "PVS pred median [IQR]": fmt_median_iqr(g["pvs_pred"]) if "pvs_pred" in g.columns else "NA",
            "PVS joint median [IQR]": fmt_median_iqr(g["pvs_joint"]) if "pvs_joint" in g.columns else "NA",
            "Divergence count summary": divergence_summary(g),
            "R-hat warnings > 1.01": rhat_warnings(g),
            "ESS warnings < 200": ess_warnings(g),
        })

    summary = pd.DataFrame(rows)

    csv_path = TABLE_DIR / f"{label}.csv"
    md_path = TABLE_DIR / f"{label}.md"
    tex_path = TABLE_DIR / f"{label}.tex"

    summary.to_csv(csv_path, index=False)

    # Markdown sem depender de tabulate
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(summary.to_markdown(index=False) if hasattr(summary, "to_markdown") else "")
    
    # Fallback manual caso tabulate não esteja instalado
    if md_path.read_text(encoding="utf-8").strip() == "":
        cols = list(summary.columns)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("| " + " | ".join(cols) + " |\n")
            f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
            for _, row in summary.iterrows():
                f.write("| " + " | ".join(str(row[c]) for c in cols) + " |\n")

    summary.to_latex(
        tex_path,
        index=False,
        escape=True,
        caption="Synthetic stress-test summary across Monte Carlo simulations.",
        label=f"tab:{label}",
    )

    print(f"\nGerado: {csv_path}")
    print(f"Gerado: {md_path}")
    print(f"Gerado: {tex_path}")

    return summary

# Tabela principal com todas as 150 simulações
summary_all = make_summary(df, "table_stress_test_summary")

# Tabela de sensibilidade: apenas IDs em que o PVS-aware teve zero divergências
pvs = df[df["model_name"] == "PVS_aware_area_MCMC"].copy()
clean_ids = set(
    pvs.loc[
        pd.to_numeric(pvs["n_divergences"], errors="coerce").fillna(0) == 0,
        "simulation_id"
    ]
)

df_clean = df[df["simulation_id"].isin(clean_ids)].copy()
summary_clean = make_summary(df_clean, "table_stress_test_summary_clean_pvs")

print("\nResumo principal:")
print(summary_all.to_string(index=False))

print("\nResumo de sensibilidade, apenas simulações com PVS-aware sem divergências:")
print(summary_clean.to_string(index=False))
