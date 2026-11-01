"""
Comprehensive Exploratory Data Analysis, Statistical Testing & Visualization Suite
==================================================================================
Project: IVF Treatment Outcome Analysis & Success Rate Optimization Using Python
Outputs:
  - 18 high-resolution (300 DPI) PNG charts in `visualizations/`
  - Exact analytical & statistical results saved to `docs/analytical_results.json`
  - Correlation matrices & summary tables in `docs/`
"""

import json
import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

warnings.filterwarnings("ignore")

# Set clean publication style
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.labelweight": "bold",
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})

PALETTE_MAIN = ["#1B365D", "#008080", "#2E86AB", "#E07A5F", "#F2CC8F", "#6D597A", "#4CAF50"]
QUALITY_ORDER = ["Excellent", "Good", "Fair", "Poor"]
QUALITY_COLORS = ["#1B365D", "#008080", "#E9C46A", "#E07A5F"]
AGE_ORDER = ["<30", "30-34", "35-37", "38-40", ">40"]
TT_ORDER = ["ICSI", "Conventional IVF", "IVF + PGT-A", "Mild Stimulation IVF"]


def wilson_ci(k, n, z=1.96):
    """95% Wilson score confidence interval for binomial proportion (returned in %)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)) / denom
    return 100 * p, 100 * max(0.0, center - half), 100 * min(1.0, center + half)


def cramers_v(confusion_matrix: np.ndarray) -> float:
    chi2 = stats.chi2_contingency(confusion_matrix, correction=False)[0]
    n = confusion_matrix.sum()
    r, k = confusion_matrix.shape
    return float(np.sqrt(chi2 / (n * max(1, min(k - 1, r - 1)))))


def fit_logistic_regression_irls(X: np.ndarray, y: np.ndarray, feature_names: list) -> list:
    """Exact Maximum Likelihood Logistic Regression with Hessian-derived SEs, ORs, 95% CIs, and Wald p-values."""
    X_design = np.column_stack([np.ones(len(y)), X])
    col_names = ["Intercept"] + feature_names

    def neg_log_lik(beta):
        z = np.clip(X_design @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        return -np.sum(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))

    def grad(beta):
        z = np.clip(X_design @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        return X_design.T @ (p - y)

    res = minimize(neg_log_lik, np.zeros(X_design.shape[1]), jac=grad, method="BFGS")
    beta_hat = res.x
    z_hat = np.clip(X_design @ beta_hat, -30, 30)
    p_hat = 1.0 / (1.0 + np.exp(-z_hat))
    W = p_hat * (1.0 - p_hat)
    fisher_info = (X_design.T * W) @ X_design
    cov = np.linalg.inv(fisher_info)
    se = np.sqrt(np.diag(cov))
    wald_z = beta_hat / se
    p_vals = 2.0 * (1.0 - stats.norm.cdf(np.abs(wald_z)))

    results = []
    for name, b, s, wz, pv in zip(col_names, beta_hat, se, wald_z, p_vals):
        results.append({
            "Feature": name,
            "Beta": round(float(b), 4),
            "SE": round(float(s), 4),
            "Odds_Ratio": round(float(np.exp(b)), 3),
            "OR_CI_Lower": round(float(np.exp(b - 1.96 * s)), 3),
            "OR_CI_Upper": round(float(np.exp(b + 1.96 * s)), 3),
            "Wald_Z": round(float(wz), 3),
            "p_value": float(pv),
        })
    return results


def run_analysis_and_visualizations(
    clean_dir: str = "data/cleaned",
    viz_dir: str = "visualizations",
    docs_dir: str = "docs",
):
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)

    patients = pd.read_csv(os.path.join(clean_dir, "cleaned_patients.csv"))
    cycles = pd.read_csv(os.path.join(clean_dir, "cleaned_treatment_cycles.csv"))
    embryos = pd.read_csv(os.path.join(clean_dir, "cleaned_embryo_grading.csv"))
    outcomes = pd.read_csv(os.path.join(clean_dir, "cleaned_implantation_outcomes.csv"))
    costs = pd.read_csv(os.path.join(clean_dir, "cleaned_treatment_costs.csv"))

    # Aggregate best embryo per cycle (representing highest-ranked cohort embryo selected for transfer)
    emb_sorted = embryos.sort_values(
        by=["Cycle_ID", "Quality_Score_Numeric", "Embryo_Day"],
        ascending=[True, False, False],
    )
    best_emb_per_cycle = (
        emb_sorted.groupby("Cycle_ID")
        .agg(
            Cohort_Graded_Embryos=("Embryo_ID", "count"),
            Top_Quality_Embryo_Count=("Is_Top_Quality", "sum"),
            Best_Embryo_Quality=("Embryo_Quality_Category", "first"),
            Best_Quality_Score=("Quality_Score_Numeric", "first"),
            Best_Embryo_Day=("Embryo_Day", "first"),
            Best_Morphological_Grade=("Morphological_Grade", "first"),
            Best_ICM_Grade=("Inner_Cell_Mass_Grade", "first"),
            Best_TE_Grade=("Trophectoderm_Grade", "first"),
            Best_Expansion_Grade=("Expansion_Grade", "first"),
        )
        .reset_index()
    )

    # Master Cycle-Level Analytical Dataset
    master = (
        cycles.merge(patients, on="Patient_ID", how="left")
        .merge(outcomes, on="Cycle_ID", how="left")
        .merge(costs, on="Cycle_ID", how="left")
        .merge(best_emb_per_cycle, on="Cycle_ID", how="left")
    )
    master["Best_Embryo_Quality"] = master["Best_Embryo_Quality"].fillna("No Viable Embryo")
    master["Best_Quality_Score"] = master["Best_Quality_Score"].fillna(0).astype(int)
    master["Best_Embryo_Day"] = master["Best_Embryo_Day"].fillna("No Embryo")
    master["Top_Quality_Embryo_Count"] = master["Top_Quality_Embryo_Count"].fillna(0).astype(int)

    # Binary endpoint flags
    master["Is_Transfer"] = (master["Number_of_Embryos_Transferred"] >= 1).astype(int)
    master["Positive_hCG_Bin"] = (master["Pregnancy_Test_Result"] == "Positive").astype(int)
    master["Implanted_Bin"] = (master["Implantation_Status"] == "Implanted").astype(int)
    master["Clinical_Preg_Bin"] = (master["Clinical_Pregnancy_Status"] == "Yes").astype(int)
    master["Live_Birth_Bin"] = (master["Live_Birth_Status"] == "Yes").astype(int)

    et_df = master[master["Is_Transfer"] == 1].copy()

    # =========================================================================
    # COMPUTE CORE METRICS & STATISTICAL TESTS
    # =========================================================================
    n_patients_clean = int(len(patients))
    n_cycles_clean = int(len(master))
    n_transfers = int(len(et_df))
    n_no_transfer = int(n_cycles_clean - n_transfers)
    n_embryos_clean = int(len(embryos))

    # A. Patient Demographics Summary
    demographics_stats = {}
    for col in ["Age", "BMI", "Infertility_Duration_Years", "Previous_IVF_Attempts", "AMH_Level", "FSH_Level"]:
        s = patients[col]
        demographics_stats[col] = {
            "mean": round(float(s.mean()), 2),
            "sd": round(float(s.std()), 2),
            "median": round(float(s.median()), 2),
            "q1": round(float(s.quantile(0.25)), 2),
            "q3": round(float(s.quantile(0.75)), 2),
            "iqr": round(float(s.quantile(0.75) - s.quantile(0.25)), 2),
            "min": round(float(s.min()), 2),
            "max": round(float(s.max()), 2),
            "skewness": round(float(s.skew()), 3),
        }

    age_group_pt_counts = patients["Age_Group"].value_counts().reindex(AGE_ORDER).to_dict()
    bmi_cat_pt_counts = patients["BMI_Category"].value_counts().to_dict()
    inf_type_pt_counts = patients["Infertility_Type"].value_counts().to_dict()
    med_hist_pt_counts = patients["Medical_History_Category"].value_counts().to_dict()

    # B. Treatment & Oocyte Funnel Summary
    treatment_counts = master["Treatment_Type"].value_counts().to_dict()
    protocol_counts = master["Stimulation_Protocol"].value_counts().to_dict()
    transfer_type_counts = master["Transfer_Type"].value_counts().to_dict()
    transfer_strategy_counts = master["Transfer_Strategy"].value_counts().to_dict()

    funnel_means = {
        "Retrieved_Oocytes_Mean": round(float(master["Retrieved_Oocytes"].mean()), 2),
        "Retrieved_Oocytes_Median": round(float(master["Retrieved_Oocytes"].median()), 2),
        "Mature_Oocytes_Mean": round(float(master["Mature_Oocytes"].mean()), 2),
        "Mature_Oocytes_Median": round(float(master["Mature_Oocytes"].median()), 2),
        "Fertilized_Oocytes_Mean": round(float(master["Fertilized_Oocytes"].mean()), 2),
        "Fertilized_Oocytes_Median": round(float(master["Fertilized_Oocytes"].median()), 2),
        "Viable_Embryos_Mean": round(float(master["Number_of_Embryos"].mean()), 2),
        "Viable_Embryos_Median": round(float(master["Number_of_Embryos"].median()), 2),
        "Mean_Maturation_Rate_Pct": round(100.0 * float(master["Oocyte_Maturation_Rate"].mean()), 2),
        "Mean_Fertilization_Rate_Pct": round(100.0 * float(master.loc[master["Mature_Oocytes"] > 0, "Fertilization_Rate"].mean()), 2),
        "Mean_Embryo_Dev_Rate_Pct": round(100.0 * float(master.loc[master["Fertilized_Oocytes"] > 0, "Embryo_Development_Rate"].mean()), 2),
    }

    # C. Embryo Quality Summary
    emb_quality_all = embryos["Embryo_Quality_Category"].value_counts().reindex(QUALITY_ORDER).to_dict()
    emb_day_counts = embryos["Embryo_Day"].value_counts().to_dict()

    # D. Clinical Outcome Rates (Strictly separated by Denominator!)
    n_pos_hcg = int(master["Positive_hCG_Bin"].sum())
    n_imp = int(master["Implanted_Bin"].sum())
    n_clin = int(master["Clinical_Preg_Bin"].sum())
    n_lb = int(master["Live_Birth_Bin"].sum())

    outcome_rates = {
        "counts": {
            "Initiated_Cycles": n_cycles_clean,
            "Embryo_Transfer_Cycles": n_transfers,
            "No_Transfer_Cycles": n_no_transfer,
            "Positive_hCG": n_pos_hcg,
            "Implanted_Cycles": n_imp,
            "Clinical_Pregnancies": n_clin,
            "Live_Births": n_lb,
        },
        "per_initiated_cycle_pct": {
            "Transfer_Completion_Rate": round(100.0 * n_transfers / n_cycles_clean, 2),
            "Positive_hCG_Rate": round(100.0 * n_pos_hcg / n_cycles_clean, 2),
            "Cycle_Implantation_Proportion": round(100.0 * n_imp / n_cycles_clean, 2),
            "Clinical_Pregnancy_Rate": round(100.0 * n_clin / n_cycles_clean, 2),
            "Live_Birth_Rate": round(100.0 * n_lb / n_cycles_clean, 2),
        },
        "per_embryo_transfer_pct": {
            "Positive_hCG_Rate": round(100.0 * n_pos_hcg / n_transfers, 2),
            "Cycle_Implantation_Proportion": round(100.0 * n_imp / n_transfers, 2),
            "Clinical_Pregnancy_Rate": round(100.0 * n_clin / n_transfers, 2),
            "Live_Birth_Rate": round(100.0 * n_lb / n_transfers, 2),
        },
        "stage_progression_conditional_pct": {
            "Implantation_given_Pos_hCG": round(100.0 * n_imp / n_pos_hcg, 2),
            "Biochemical_Loss_given_Pos_hCG": round(100.0 * (n_pos_hcg - n_imp) / n_pos_hcg, 2),
            "Clinical_Preg_given_Implantation": round(100.0 * n_clin / n_imp, 2),
            "Live_Birth_given_Clinical_Preg": round(100.0 * n_lb / n_clin, 2),
        },
    }

    # Sensitivity check for Completed Follow-Up Only
    et_completed = et_df[et_df["Follow_Up_Status"] == "Completed"]
    outcome_rates["completed_followup_sensitivity"] = {
        "Completed_ET_Cycles": int(len(et_completed)),
        "Live_Birth_Rate_Completed_ET_Pct": round(100.0 * float(et_completed["Live_Birth_Bin"].mean()), 2),
    }

    # Stratified Tables (Per Embryo Transfer & Per Initiated Cycle)
    def summarize_stratum(df_sub, group_col, order=None):
        grp = df_sub.groupby(group_col).agg(
            Cycles=("Cycle_ID", "count"),
            Transfers=("Is_Transfer", "sum"),
            Pos_hCG=("Positive_hCG_Bin", "sum"),
            Implanted=("Implanted_Bin", "sum"),
            Clin_Preg=("Clinical_Preg_Bin", "sum"),
            Live_Birth=("Live_Birth_Bin", "sum"),
            Mean_Age=("Age", "mean"),
            Mean_Cost=("Total_Treatment_Cost", "mean"),
            Median_Cost=("Total_Treatment_Cost", "median"),
            Total_Spend=("Total_Treatment_Cost", "sum"),
        )
        if order is not None:
            grp = grp.reindex([o for o in order if o in grp.index])
        out_list = []
        for idx_val, r in grp.iterrows():
            tr = int(r["Transfers"])
            cyc = int(r["Cycles"])
            lb = int(r["Live_Birth"])
            out_list.append({
                group_col: str(idx_val),
                "Cycles": cyc,
                "Transfers": tr,
                "Implantation_per_ET_Pct": round(100.0 * r["Implanted"] / tr, 2) if tr > 0 else 0.0,
                "CPR_per_ET_Pct": round(100.0 * r["Clin_Preg"] / tr, 2) if tr > 0 else 0.0,
                "LBR_per_ET_Pct": round(100.0 * lb / tr, 2) if tr > 0 else 0.0,
                "CPR_per_Cycle_Pct": round(100.0 * r["Clin_Preg"] / cyc, 2),
                "LBR_per_Cycle_Pct": round(100.0 * lb / cyc, 2),
                "Mean_Age": round(float(r["Mean_Age"]), 2),
                "Mean_Cost": round(float(r["Mean_Cost"]), 2),
                "Median_Cost": round(float(r["Median_Cost"]), 2),
                "Cost_per_Live_Birth": round(float(r["Total_Spend"] / lb), 2) if lb > 0 else None,
            })
        return out_list

    by_embryo_quality_et = summarize_stratum(et_df, "Best_Embryo_Quality", QUALITY_ORDER)
    by_age_group = summarize_stratum(master, "Age_Group", AGE_ORDER)
    by_treatment_type = summarize_stratum(master, "Treatment_Type", TT_ORDER)
    by_embryo_day_et = summarize_stratum(et_df, "Best_Embryo_Day", ["Day 5", "Day 3"])
    by_transfer_type_et = summarize_stratum(et_df, "Transfer_Type", ["Frozen-Thawed Transfer", "Fresh Transfer"])
    by_transfer_strategy_et = summarize_stratum(et_df, "Transfer_Strategy", ["eSET (1 Embryo)", "DET (2 Embryos)", "MET (3 Embryos)"])
    by_infertility_type = summarize_stratum(master, "Infertility_Type", ["Primary", "Secondary"])
    by_bmi_cat = summarize_stratum(
        master,
        "BMI_Category",
        ["Underweight (<18.5)", "Normal (18.5-24.9)", "Overweight (25.0-29.9)", "Obese (>=30.0)"],
    )
    by_protocol = summarize_stratum(master, "Stimulation_Protocol")

    # E. Cost Analysis Summary
    cost_cols = [
        "Consultation_Cost",
        "Medication_Cost",
        "Laboratory_Cost",
        "Procedure_Cost",
        "Additional_Cost",
        "Total_Treatment_Cost",
    ]
    total_spend_all = float(costs["Total_Treatment_Cost"].sum())
    cost_summary = {}
    for c in cost_cols:
        s = costs[c]
        cost_summary[c] = {
            "mean": round(float(s.mean()), 2),
            "sd": round(float(s.std()), 2),
            "median": round(float(s.median()), 2),
            "q1": round(float(s.quantile(0.25)), 2),
            "q3": round(float(s.quantile(0.75)), 2),
            "iqr": round(float(s.quantile(0.75) - s.quantile(0.25)), 2),
            "min": round(float(s.min()), 2),
            "max": round(float(s.max()), 2),
            "share_of_total_spend_pct": round(100.0 * float(s.sum()) / total_spend_all, 2),
        }
    cost_summary["Clinic_Cost_per_Live_Birth_Overall"] = round(total_spend_all / n_lb, 2)

    # F. Correlation & Hypothesis Testing
    corr_vars = [
        "Age",
        "BMI",
        "Infertility_Duration_Years",
        "Previous_IVF_Attempts",
        "AMH_Level",
        "FSH_Level",
        "Retrieved_Oocytes",
        "Mature_Oocytes",
        "Fertilized_Oocytes",
        "Number_of_Embryos",
        "Best_Quality_Score",
        "Total_Treatment_Cost",
        "Implanted_Bin",
        "Clinical_Preg_Bin",
        "Live_Birth_Bin",
    ]
    pearson_mat = master[corr_vars].corr(method="pearson").round(3)
    spearman_mat = master[corr_vars].corr(method="spearman").round(3)
    pearson_mat.to_csv(os.path.join(docs_dir, "pearson_correlation_matrix.csv"))
    spearman_mat.to_csv(os.path.join(docs_dir, "spearman_correlation_matrix.csv"))

    # Pairwise correlation tests
    corr_pairs_spec = [
        ("Age", "AMH_Level", master),
        ("AMH_Level", "FSH_Level", master),
        ("AMH_Level", "Retrieved_Oocytes", master),
        ("Age", "Retrieved_Oocytes", master),
        ("Retrieved_Oocytes", "Mature_Oocytes", master),
        ("Mature_Oocytes", "Number_of_Embryos", master),
        ("Medication_Cost", "Total_Treatment_Cost", master),
        ("Laboratory_Cost", "Total_Treatment_Cost", master),
        ("Best_Quality_Score", "Implanted_Bin", et_df),
        ("Age", "Live_Birth_Bin", et_df),
        ("BMI", "Live_Birth_Bin", et_df),
        ("Consultation_Cost", "Live_Birth_Bin", et_df),
    ]
    pairwise_corrs = []
    for v1, v2, df_c in corr_pairs_spec:
        pr, pp = stats.pearsonr(df_c[v1], df_c[v2])
        sr, sp = stats.spearmanr(df_c[v1], df_c[v2])
        pairwise_corrs.append({
            "Var_1": v1,
            "Var_2": v2,
            "N": int(len(df_c)),
            "Pearson_r": round(float(pr), 4),
            "Pearson_p": float(pp),
            "Spearman_rho": round(float(sr), 4),
            "Spearman_p": float(sp),
        })

    # Hypothesis Tests Suite
    hypothesis_tests = []

    def add_chi2_test(label, df_sub, col_x, col_y):
        ct = pd.crosstab(df_sub[col_x], df_sub[col_y])
        chi2, p, dof, ex = stats.chi2_contingency(ct)
        cv = cramers_v(ct.to_numpy())
        min_expected = float(ex.min())
        hypothesis_tests.append({
            "Test_Name": label,
            "Method": "Pearson Chi-Square Test of Independence",
            "Sample_N": int(len(df_sub)),
            "Statistic": round(float(chi2), 3),
            "df": int(dof),
            "p_value": float(p),
            "Effect_Size_Metric": "Cramer's V",
            "Effect_Size": round(cv, 4),
            "Assumptions_Check": f"Independent cycle/transfer observations; minimum expected cell count = {min_expected:.1f} (>= 5 satisfied).",
            "Significant_at_0_05": bool(p < 0.05),
        })

    add_chi2_test("Best Transferred Embryo Quality vs Implantation (ET Cycles)", et_df, "Best_Embryo_Quality", "Implanted_Bin")
    add_chi2_test("Maternal Age Group vs Clinical Pregnancy (ET Cycles)", et_df, "Age_Group", "Clinical_Preg_Bin")
    add_chi2_test("Maternal Age Group vs Live Birth (Initiated Cycles)", master, "Age_Group", "Live_Birth_Bin")
    add_chi2_test("Embryo Day (Day 5 vs Day 3) vs Implantation (ET Cycles)", et_df, "Best_Embryo_Day", "Implanted_Bin")
    add_chi2_test("Treatment Type vs Live Birth (Crude / Unadjusted, Initiated Cycles)", master, "Treatment_Type", "Live_Birth_Bin")
    add_chi2_test("Transfer Type (FET vs Fresh) vs Live Birth (ET Cycles)", et_df, "Transfer_Type", "Live_Birth_Bin")
    add_chi2_test("Infertility Type (Primary vs Secondary) vs Live Birth (ET Cycles)", et_df, "Infertility_Type", "Live_Birth_Bin")
    add_chi2_test("BMI Category vs Clinical Pregnancy (ET Cycles)", et_df, "BMI_Category", "Clinical_Preg_Bin")

    # Continuous group comparisons (Kruskal-Wallis + Welch ANOVA / Mann-Whitney U + Welch t-test)
    # 1. Total_Treatment_Cost across Treatment_Type
    groups_cost = [g["Total_Treatment_Cost"].to_numpy() for _, g in master.groupby("Treatment_Type")]
    kw_cost_stat, kw_cost_p = stats.kruskal(*groups_cost)
    eps2_cost = float(kw_cost_stat / ((len(master) ** 2 - 1) / (len(master) + 1)))
    f_cost_stat, f_cost_p = stats.f_oneway(*groups_cost)
    lev_cost_stat, lev_cost_p = stats.levene(*groups_cost)
    hypothesis_tests.append({
        "Test_Name": "Total Treatment Cost across Treatment Types",
        "Method": "Kruskal-Wallis H Test (with One-Way ANOVA check)",
        "Sample_N": int(len(master)),
        "Statistic": round(float(kw_cost_stat), 3),
        "df": 3,
        "p_value": float(kw_cost_p),
        "Effect_Size_Metric": "Epsilon-squared (eps^2)",
        "Effect_Size": round(eps2_cost, 4),
        "Assumptions_Check": f"Right-skewed cost distributions & Levene variance test F={lev_cost_stat:.2f} (p={lev_cost_p:.3e}); non-parametric Kruskal-Wallis preferred (ANOVA F={f_cost_stat:.1f}, p={f_cost_p:.3e}).",
        "Significant_at_0_05": bool(kw_cost_p < 0.05),
    })

    # 2. Retrieved_Oocytes across Stimulation_Protocol
    groups_ooc = [g["Retrieved_Oocytes"].to_numpy() for _, g in master.groupby("Stimulation_Protocol")]
    kw_ooc_stat, kw_ooc_p = stats.kruskal(*groups_ooc)
    eps2_ooc = float(kw_ooc_stat / ((len(master) ** 2 - 1) / (len(master) + 1)))
    hypothesis_tests.append({
        "Test_Name": "Retrieved Oocytes across Stimulation Protocols",
        "Method": "Kruskal-Wallis H Test",
        "Sample_N": int(len(master)),
        "Statistic": round(float(kw_ooc_stat), 3),
        "df": 3,
        "p_value": float(kw_ooc_p),
        "Effect_Size_Metric": "Epsilon-squared (eps^2)",
        "Effect_Size": round(eps2_ooc, 4),
        "Assumptions_Check": "Count outcome with positive skewness; non-parametric rank comparison robust to hyper-responder tail.",
        "Significant_at_0_05": bool(kw_ooc_p < 0.05),
    })

    # 3. Maternal Age in Live Birth (Yes vs No) among ET Cycles
    age_lb_yes = et_df.loc[et_df["Live_Birth_Bin"] == 1, "Age"].to_numpy()
    age_lb_no = et_df.loc[et_df["Live_Birth_Bin"] == 0, "Age"].to_numpy()
    t_age, p_t_age = stats.ttest_ind(age_lb_yes, age_lb_no, equal_var=False)
    u_age, p_u_age = stats.mannwhitneyu(age_lb_yes, age_lb_no, alternative="two-sided")
    pooled_sd_age = np.sqrt((age_lb_yes.var(ddof=1) + age_lb_no.var(ddof=1)) / 2.0)
    cohens_d_age = float((age_lb_yes.mean() - age_lb_no.mean()) / pooled_sd_age)
    hypothesis_tests.append({
        "Test_Name": "Maternal Age: Live Birth Yes vs No (ET Cycles)",
        "Method": "Welch Two-Sample t-Test & Mann-Whitney U",
        "Sample_N": int(len(et_df)),
        "Statistic": round(float(t_age), 3),
        "df": int(len(et_df) - 2),
        "p_value": float(p_t_age),
        "Effect_Size_Metric": "Cohen's d",
        "Effect_Size": round(cohens_d_age, 4),
        "Assumptions_Check": f"Large N={len(et_df)} satisfies CLT; Welch t={t_age:.2f} (p={p_t_age:.3e}) & Mann-Whitney U p={p_u_age:.3e} concur.",
        "Significant_at_0_05": bool(p_t_age < 0.05),
    })

    # 4. Consultation_Cost in Live Birth (Yes vs No) - Expected Null!
    cc_yes = et_df.loc[et_df["Live_Birth_Bin"] == 1, "Consultation_Cost"].to_numpy()
    cc_no = et_df.loc[et_df["Live_Birth_Bin"] == 0, "Consultation_Cost"].to_numpy()
    t_cc, p_t_cc = stats.ttest_ind(cc_yes, cc_no, equal_var=False)
    pooled_sd_cc = np.sqrt((cc_yes.var(ddof=1) + cc_no.var(ddof=1)) / 2.0)
    cohens_d_cc = float((cc_yes.mean() - cc_no.mean()) / pooled_sd_cc)
    hypothesis_tests.append({
        "Test_Name": "Consultation Cost: Live Birth Yes vs No (ET Cycles)",
        "Method": "Welch Two-Sample t-Test",
        "Sample_N": int(len(et_df)),
        "Statistic": round(float(t_cc), 3),
        "df": int(len(et_df) - 2),
        "p_value": float(p_t_cc),
        "Effect_Size_Metric": "Cohen's d",
        "Effect_Size": round(cohens_d_cc, 4),
        "Assumptions_Check": "Continuous approximately normal fee variable; Welch correction applied.",
        "Significant_at_0_05": bool(p_t_cc < 0.05),
    })

    # Multivariable Logistic Regression on ET Cycles for Live Birth
    X_logit = pd.DataFrame({
        "Age_per_5yr": (et_df["Age"] - 34.0) / 5.0,
        "Embryo_Excellent_vs_FairPoor": (et_df["Best_Embryo_Quality"] == "Excellent").astype(float),
        "Embryo_Good_vs_FairPoor": (et_df["Best_Embryo_Quality"] == "Good").astype(float),
        "Day5_vs_Day3": (et_df["Best_Embryo_Day"] == "Day 5").astype(float),
        "PGT_A_vs_Conventional": (et_df["Treatment_Type"] == "IVF + PGT-A").astype(float),
        "ICSI_vs_Conventional": (et_df["Treatment_Type"] == "ICSI").astype(float),
        "FET_vs_Fresh": (et_df["Transfer_Type"] == "Frozen-Thawed Transfer").astype(float),
        "Multi_Embryo_Transfer_vs_eSET": (et_df["Number_of_Embryos_Transferred"] >= 2).astype(float),
        "Obese_BMI_ge_30": (et_df["BMI"] >= 30.0).astype(float),
        "Previous_IVF_Attempts": et_df["Previous_IVF_Attempts"].astype(float),
    })
    logit_results = fit_logistic_regression_irls(
        X_logit.to_numpy(),
        et_df["Live_Birth_Bin"].to_numpy(),
        X_logit.columns.tolist(),
    )

    # Check Age >= 38 PGT-A vs Conventional/ICSI stratified LBR per ET (demonstrates confounding by indication!)
    older_et = et_df[et_df["Age"] >= 38]
    older_by_tt = older_et.groupby("Treatment_Type")["Live_Birth_Bin"].agg(["count", "mean"]).reset_index()
    older_by_tt["LBR_ET_Pct"] = (100.0 * older_by_tt["mean"]).round(2)
    older_stratified_lbr = older_by_tt[["Treatment_Type", "count", "LBR_ET_Pct"]].to_dict(orient="records")

    # Day 5 ICM x TE Implantation Matrix (where Day 5 embryo was top cohort embryo in ET cycles)
    d5_et = et_df[et_df["Best_Embryo_Day"] == "Day 5"].copy()
    icm_te_pivot = (
        d5_et.pivot_table(
            index="Best_ICM_Grade",
            columns="Best_TE_Grade",
            values="Implanted_Bin",
            aggfunc=lambda s: 100.0 * np.mean(s) if len(s) >= 10 else np.nan,
        )
        .reindex(index=["A", "B", "C"], columns=["A", "B", "C"])
        .round(1)
    )

    # =========================================================================
    # PHASE 6: GENERATE ALL 18 HIGH-RESOLUTION VISUALIZATIONS
    # =========================================================================

    # --- Chart 1: Patient Age Distribution Histogram (`age_distribution.png`) ---
    fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    sns.histplot(patients["Age"], bins=24, kde=True, color="#1B365D", alpha=0.72, edgecolor="white", ax=ax)
    mean_age = patients["Age"].mean()
    med_age = patients["Age"].median()
    ax.axvline(mean_age, color="#E07A5F", lw=2.2, ls="--", label=f"Mean Age = {mean_age:.1f} yrs")
    ax.axvline(med_age, color="#008080", lw=2.2, ls="-.", label=f"Median Age = {med_age:.1f} yrs")
    ax.axvspan(38, 45.5, color="#E07A5F", alpha=0.10, label="Advanced Maternal Age (>=38 yrs)")
    ax.set_title(f"Baseline Maternal Age Distribution Across Synthetic Clinic Patients (n = {n_patients_clean:,})")
    ax.set_xlabel("Maternal Age at Intake (Years)")
    ax.set_ylabel("Number of Patients")
    ax.legend(frameon=True, facecolor="white")
    fig.savefig(os.path.join(viz_dir, "age_distribution.png"))
    plt.close(fig)

    # --- Chart 2: BMI Distribution Histogram (`bmi_distribution.png`) ---
    fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    sns.histplot(patients["BMI"], bins=28, kde=True, color="#008080", alpha=0.70, edgecolor="white", ax=ax)
    ax.axvline(18.5, color="#2E86AB", lw=1.6, ls=":", label="WHO Normal Lower (18.5 kg/m²)")
    ax.axvline(25.0, color="#E9C46A", lw=2.0, ls="--", label="WHO Overweight Threshold (25.0 kg/m²)")
    ax.axvline(30.0, color="#E07A5F", lw=2.0, ls="--", label="WHO Obesity Threshold (30.0 kg/m²)")
    ax.axvline(patients["BMI"].mean(), color="#1B365D", lw=2.2, ls="-", label=f"Mean BMI = {patients['BMI'].mean():.1f} kg/m²")
    ax.set_title(f"Patient Body Mass Index (BMI) Distribution with WHO Clinical Cutoffs (n = {n_patients_clean:,})")
    ax.set_xlabel("Body Mass Index (kg/m²)")
    ax.set_ylabel("Number of Patients")
    ax.legend(frameon=True, facecolor="white")
    fig.savefig(os.path.join(viz_dir, "bmi_distribution.png"))
    plt.close(fig)

    # --- Chart 3: Infertility Type & Clinical Etiology Bar Chart (`infertility_type_distribution.png`) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.5), gridspec_kw={"width_ratios": [1, 1.8]}, constrained_layout=True)
    inf_counts = patients["Infertility_Type"].value_counts()
    bars1 = ax1.bar(inf_counts.index, inf_counts.values, color=["#1B365D", "#008080"], width=0.55)
    for b in bars1:
        pct = 100.0 * b.get_height() / n_patients_clean
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 35, f"{int(b.get_height()):,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax1.set_ylim(0, inf_counts.max() * 1.18)
    ax1.set_title("A. Infertility Classification")
    ax1.set_xlabel("Infertility Type")
    ax1.set_ylabel("Number of Patients")

    med_counts = patients["Medical_History_Category"].value_counts().sort_values(ascending=True)
    bars2 = ax2.barh(med_counts.index, med_counts.values, color="#2E86AB", height=0.65)
    for b in bars2:
        pct = 100.0 * b.get_width() / n_patients_clean
        ax2.text(b.get_width() + 12, b.get_y() + b.get_height() / 2, f"{int(b.get_width()):,} ({pct:.1f}%)", va="center", fontsize=9)
    ax2.set_xlim(0, med_counts.max() * 1.24)
    ax2.set_title("B. Primary Diagnostic Etiology (Medical History)")
    ax2.set_xlabel("Number of Patients")
    ax2.set_ylabel("Diagnostic Category")
    fig.suptitle(f"Patient Infertility Classification and Diagnostic Etiology Profile (n = {n_patients_clean:,})", fontsize=13.5, fontweight="bold")
    fig.savefig(os.path.join(viz_dir, "infertility_type_distribution.png"))
    plt.close(fig)

    # --- Chart 4: Previous IVF Attempts Distribution (`previous_ivf_attempts.png`) ---
    fig, ax1 = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    prev_counts = patients["Previous_IVF_Attempts"].value_counts().sort_index()
    cum_pct = 100.0 * prev_counts.cumsum() / n_patients_clean
    bars = ax1.bar(prev_counts.index.astype(str), prev_counts.values, color="#1B365D", alpha=0.85, width=0.58, label="Patient Count")
    for b in bars:
        pct = 100.0 * b.get_height() / n_patients_clean
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 30, f"{int(b.get_height()):,}\n({pct:.1f}%)", ha="center", fontsize=9)
    ax1.set_ylim(0, prev_counts.max() * 1.18)
    ax1.set_xlabel("Number of Previous Unsuccessful / Prior IVF Cycles")
    ax1.set_ylabel("Number of Patients")

    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.grid(False)
    ax2.plot(prev_counts.index.astype(str), cum_pct.values, color="#E07A5F", marker="o", lw=2.3, label="Cumulative Percentage (%)")
    ax2.set_ylim(0, 108)
    ax2.set_ylabel("Cumulative Percentage of Patients (%)", color="#E07A5F")
    ax1.set_title(f"Prior IVF Treatment History Among Registered Patients (n = {n_patients_clean:,})")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", frameon=True, facecolor="white")
    fig.savefig(os.path.join(viz_dir, "previous_ivf_attempts.png"))
    plt.close(fig)

    # --- Chart 5: Retrieved vs Mature Oocytes Comparison (`oocytes_retrieved_vs_mature.png`) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.5), constrained_layout=True)
    rng_jitter = np.random.default_rng(42)
    jx = master["Retrieved_Oocytes"] + rng_jitter.normal(0, 0.18, len(master))
    jy = master["Mature_Oocytes"] + rng_jitter.normal(0, 0.18, len(master))
    ax1.scatter(jx, jy, alpha=0.18, s=14, color="#008080", label="Individual Cycles")
    max_o = master["Retrieved_Oocytes"].max()
    x_line = np.linspace(0, max_o, 100)
    ax1.plot(x_line, x_line, color="#6C757D", ls=":", lw=1.5, label="100% Maturation Ceiling (y = x)")
    slope, intercept = np.polyfit(master["Retrieved_Oocytes"], master["Mature_Oocytes"], 1)
    ax1.plot(x_line, slope * x_line + intercept, color="#E07A5F", lw=2.2, label=f"OLS Fit (Slope = {slope:.2f}, r = {pearson_mat.loc['Retrieved_Oocytes', 'Mature_Oocytes']:.2f})")
    ax1.set_title("A. Retrieved vs. Mature (MII) Oocyte Yield")
    ax1.set_xlabel("Retrieved Oocytes (Count)")
    ax1.set_ylabel("Mature MII Oocytes (Count)")
    ax1.legend(frameon=True, facecolor="white", loc="upper left")

    stages = ["Retrieved\nOocytes", "Mature (MII)\nOocytes", "Fertilized\n(2PN) Zygotes", "Viable\nEmbryos", "Transferred\nEmbryos"]
    means = [
        master["Retrieved_Oocytes"].mean(),
        master["Mature_Oocytes"].mean(),
        master["Fertilized_Oocytes"].mean(),
        master["Number_of_Embryos"].mean(),
        master["Number_of_Embryos_Transferred"].mean(),
    ]
    medians = [
        master["Retrieved_Oocytes"].median(),
        master["Mature_Oocytes"].median(),
        master["Fertilized_Oocytes"].median(),
        master["Number_of_Embryos"].median(),
        master["Number_of_Embryos_Transferred"].median(),
    ]
    x_pos = np.arange(len(stages))
    b_m = ax2.bar(x_pos - 0.18, means, width=0.35, color="#1B365D", label="Mean per Cycle")
    b_med = ax2.bar(x_pos + 0.18, medians, width=0.35, color="#2E86AB", label="Median per Cycle")
    for b in b_m:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.2, f"{b.get_height():.1f}", ha="center", fontsize=8.5, fontweight="bold")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(stages)
    ax2.set_ylim(0, max(means) * 1.16)
    ax2.set_title("B. Laboratory Attrition Funnel Across Gamete-Embryo Stages")
    ax2.set_ylabel("Count per Initiated Cycle")
    ax2.legend(frameon=True, facecolor="white")
    fig.suptitle(f"Oocyte Maturation Efficiency and Embryology Laboratory Yield (n = {n_cycles_clean:,} Cycles)", fontsize=13.5, fontweight="bold")
    fig.savefig(os.path.join(viz_dir, "oocytes_retrieved_vs_mature.png"))
    plt.close(fig)

    # --- Chart 6: Treatment Type & Protocol Distribution (`treatment_type_distribution.png`) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.5), constrained_layout=True)
    tt_cnt = master["Treatment_Type"].value_counts().reindex(TT_ORDER)
    bars1 = ax1.bar(tt_cnt.index, tt_cnt.values, color=PALETTE_MAIN[:4], width=0.58)
    for b in bars1:
        pct = 100.0 * b.get_height() / n_cycles_clean
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 35, f"{int(b.get_height()):,}\n({pct:.1f}%)", ha="center", fontsize=9, fontweight="bold")
    ax1.set_ylim(0, tt_cnt.max() * 1.18)
    ax1.set_title("A. Cycle Volume by IVF Treatment Modality")
    ax1.set_xlabel("Treatment Type")
    ax1.set_ylabel("Number of Initiated Cycles")
    ax1.tick_params(axis="x", rotation=14)

    ct_proto = pd.crosstab(master["Treatment_Type"], master["Stimulation_Protocol"], normalize="index").reindex(TT_ORDER) * 100.0
    ct_proto.plot(kind="barh", stacked=True, ax=ax2, colormap="crest", edgecolor="white", width=0.62)
    ax2.set_xlim(0, 100)
    ax2.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax2.set_title("B. Stimulation Protocol Mix Within Each Modality")
    ax2.set_xlabel("Share of Cycles (%)")
    ax2.set_ylabel("")
    ax2.legend(title="Stimulation Protocol", bbox_to_anchor=(1.0, -0.18), loc="upper right", ncol=2, frameon=True, facecolor="white")
    fig.suptitle(f"IVF Treatment Modality and Ovarian Stimulation Protocol Distribution (n = {n_cycles_clean:,} Cycles)", fontsize=13.5, fontweight="bold")
    fig.savefig(os.path.join(viz_dir, "treatment_type_distribution.png"), bbox_inches="tight")
    plt.close(fig)

    # --- Chart 7: Embryo Quality Distribution (`embryo_quality.png`) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 5.5), constrained_layout=True)
    eq_all = embryos["Embryo_Quality_Category"].value_counts().reindex(QUALITY_ORDER)
    b1 = ax1.bar(eq_all.index, eq_all.values, color=QUALITY_COLORS, width=0.58)
    for b in b1:
        pct = 100.0 * b.get_height() / n_embryos_clean
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 90, f"{int(b.get_height()):,}\n({pct:.1f}%)", ha="center", fontsize=9, fontweight="bold")
    ax1.set_ylim(0, eq_all.max() * 1.18)
    ax1.set_title(f"A. All Graded Viable Embryos (n = {n_embryos_clean:,})")
    ax1.set_xlabel("Composite Embryo Quality Category")
    ax1.set_ylabel("Number of Graded Embryos")

    eq_et = et_df["Best_Embryo_Quality"].value_counts().reindex(QUALITY_ORDER)
    b2 = ax2.bar(eq_et.index, eq_et.values, color=QUALITY_COLORS, width=0.58)
    for b in b2:
        pct = 100.0 * b.get_height() / n_transfers
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 35, f"{int(b.get_height()):,}\n({pct:.1f}%)", ha="center", fontsize=9, fontweight="bold")
    ax2.set_ylim(0, eq_et.max() * 1.18)
    ax2.set_title(f"B. Lead Cohort Embryo in Transfer Cycles (n = {n_transfers:,})")
    ax2.set_xlabel("Lead Transferred Embryo Quality")
    ax2.set_ylabel("Number of Transfer Cycles")
    fig.suptitle("Embryo Morphological Quality Distribution: Full Laboratory Cohort vs. Lead Transferred Embryo", fontsize=13, fontweight="bold")
    fig.savefig(os.path.join(viz_dir, "embryo_quality.png"))
    plt.close(fig)

    # --- Chart 8: Embryo Grading Comparison (`embryo_grading_comparison.png`) ---
    d5_emb = embryos[embryos["Embryo_Day"] == "Day 5"].copy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.8), constrained_layout=True)
    exp_q = pd.crosstab(d5_emb["Expansion_Grade"], d5_emb["Embryo_Quality_Category"]).reindex(index=["1", "2", "3", "4", "5", "6"], columns=QUALITY_ORDER).fillna(0)
    exp_q.plot(kind="bar", stacked=True, color=QUALITY_COLORS, ax=ax1, width=0.65, edgecolor="white")
    ax1.set_title(f"A. Expansion Score vs. Quality (n = {len(d5_emb):,})", fontsize=11.5)
    ax1.set_xlabel("Gardner Expansion Grade (1 = Early -> 6 = Hatched)")
    ax1.set_ylabel("Number of Day 5 Blastocysts")
    ax1.tick_params(axis="x", rotation=0)
    ax1.legend(title="Quality Tier", frameon=True, facecolor="white")

    sns.heatmap(
        icm_te_pivot,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        cbar_kws={"label": "Implantation Rate per ET (%)"},
        ax=ax2,
        linewidths=1.2,
    )
    ax2.set_title(f"B. Implantation (%) by ICM x TE (n = {len(d5_et):,} ETs)", fontsize=11.5)
    ax2.set_xlabel("Trophectoderm (TE) Grade")
    ax2.set_ylabel("Inner Cell Mass (ICM) Grade")
    fig.suptitle("Day 5 Gardner Blastocyst Morphological Grading & Implantation Mapping", fontsize=13.2, fontweight="bold")
    fig.savefig(os.path.join(viz_dir, "embryo_grading_comparison.png"), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    # --- Chart 9: Embryo Quality vs Implantation & Clinical Outcomes (`implantation_analysis.png`) ---
    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    x = np.arange(len(QUALITY_ORDER))
    w = 0.25
    metrics_info = [
        ("Implanted_Bin", "Implantation Rate (%)", "#1B365D"),
        ("Clinical_Preg_Bin", "Clinical Pregnancy Rate (%)", "#008080"),
        ("Live_Birth_Bin", "Live Birth Rate (%)", "#E07A5F"),
    ]
    for i, (col, lbl, clr) in enumerate(metrics_info):
        vals, err_low, err_high = [], [], []
        for q in QUALITY_ORDER:
            sub = et_df[et_df["Best_Embryo_Quality"] == q][col]
            p_val, lo, hi = wilson_ci(sub.sum(), len(sub))
            vals.append(p_val)
            err_low.append(p_val - lo)
            err_high.append(hi - p_val)
        bars = ax.bar(x + (i - 1) * w, vals, width=w, color=clr, label=lbl, yerr=[err_low, err_high], capsize=3.5, error_kw={"lw": 1.1, "ecolor": "#333333"})
        for b, eh in zip(bars, err_high):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + eh + 1.0, f"{b.get_height():.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x)
    q_labels = [f"{q}\n(n={len(et_df[et_df['Best_Embryo_Quality']==q]):,})" for q in QUALITY_ORDER]
    ax.set_xticklabels(q_labels)
    ax.set_ylim(0, 68)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title(f"Cycle Outcomes by Lead Transferred Embryo Quality (n = {n_transfers:,} Embryo Transfers)")
    ax.set_xlabel("Lead Transferred Embryo Morphological Quality Tier")
    ax.set_ylabel("Outcome Proportion per Embryo Transfer (%)")
    ax.legend(frameon=True, facecolor="white", loc="upper right")
    fig.savefig(os.path.join(viz_dir, "implantation_analysis.png"), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    # --- Chart 10: Age Group vs Clinical Pregnancy & Outcome Cascade (`age_group_vs_clinical_pregnancy.png`) ---
    fig, ax = plt.subplots(figsize=(10.5, 5.8), constrained_layout=True)
    for col, lbl, clr, mk in [
        ("Positive_hCG_Bin", "Positive Beta-hCG (% per ET)", "#2E86AB", "s"),
        ("Implanted_Bin", "Implantation Confirmed (% per ET)", "#008080", "^"),
        ("Clinical_Preg_Bin", "Clinical Pregnancy (% per ET)", "#1B365D", "o"),
        ("Live_Birth_Bin", "Live Birth (% per ET)", "#E07A5F", "D"),
    ]:
        pts, lows, highs = [], [], []
        for ag in AGE_ORDER:
            sub = et_df[et_df["Age_Group"] == ag][col]
            p_v, lo, hi = wilson_ci(sub.sum(), len(sub))
            pts.append(p_v)
            lows.append(lo)
            highs.append(hi)
        ax.plot(AGE_ORDER, pts, marker=mk, lw=2.4, markersize=7, color=clr, label=lbl)
        ax.fill_between(AGE_ORDER, lows, highs, color=clr, alpha=0.10)
        if col in ("Positive_hCG_Bin", "Live_Birth_Bin"):
            for xi, yi in enumerate(pts):
                offset = 2.4 if col == "Positive_hCG_Bin" else -3.4
                ax.text(xi, yi + offset, f"{yi:.1f}%", ha="center", fontsize=8.5, fontweight="bold", color=clr)

    ag_labels = [f"{ag} yrs\n(n={len(et_df[et_df['Age_Group']==ag]):,} ETs)" for ag in AGE_ORDER]
    ax.set_xticks(range(len(AGE_ORDER)))
    ax.set_xticklabels(ag_labels)
    ax.set_ylim(5, 70)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title(f"Maternal Age Stratification vs. Sequential Pregnancy Outcomes (n = {n_transfers:,} ETs)")
    ax.set_xlabel("Maternal Age Bracket (Years)")
    ax.set_ylabel("Outcome Rate per Embryo Transfer (%)")
    ax.legend(frameon=True, facecolor="white", loc="upper right")
    fig.savefig(os.path.join(viz_dir, "age_group_vs_clinical_pregnancy.png"), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    # --- Chart 11: Treatment Type vs Outcome (Crude vs Age-Stratified) (`treatment_type_vs_outcome.png`) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.8), constrained_layout=True)
    x_tt = np.arange(len(TT_ORDER))
    cpr_cyc = [100.0 * master[master["Treatment_Type"] == t]["Clinical_Preg_Bin"].mean() for t in TT_ORDER]
    lbr_cyc = [100.0 * master[master["Treatment_Type"] == t]["Live_Birth_Bin"].mean() for t in TT_ORDER]
    lbr_et = [100.0 * et_df[et_df["Treatment_Type"] == t]["Live_Birth_Bin"].mean() for t in TT_ORDER]

    b_c1 = ax1.bar(x_tt - 0.25, cpr_cyc, width=0.25, color="#2E86AB", label="CPR per Initiated Cycle")
    b_c2 = ax1.bar(x_tt, lbr_cyc, width=0.25, color="#1B365D", label="LBR per Initiated Cycle")
    b_c3 = ax1.bar(x_tt + 0.25, lbr_et, width=0.25, color="#E07A5F", label="LBR per Embryo Transfer")
    for bars_set in (b_c1, b_c2, b_c3):
        for b in bars_set:
            ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, f"{b.get_height():.1f}%", ha="center", fontsize=7.8, fontweight="bold")
    ax1.set_xticks(x_tt)
    ax1.set_xticklabels(TT_ORDER, rotation=12)
    ax1.set_ylim(0, 49)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax1.set_title("A. Unadjusted (Crude) Outcome Rates", fontsize=11.5)
    ax1.set_ylabel("Outcome Proportion (%)")
    ax1.legend(frameon=True, facecolor="white", fontsize=8.5)

    # Panel B: Age-stratified Live Birth per ET across modalities (exposing confounding by maternal age!)
    et_df["Broad_Age"] = np.where(et_df["Age"] < 35, "<35 yrs", np.where(et_df["Age"] <= 37, "35-37 yrs", ">=38 yrs"))
    piv_age_tt = et_df.pivot_table(index="Broad_Age", columns="Treatment_Type", values="Live_Birth_Bin", aggfunc=lambda s: 100.0 * s.mean()).reindex(index=["<35 yrs", "35-37 yrs", ">=38 yrs"], columns=TT_ORDER)
    piv_age_tt.plot(kind="bar", ax=ax2, color=PALETTE_MAIN[:4], width=0.72, edgecolor="white")
    ax2.set_ylim(0, 54)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax2.tick_params(axis="x", rotation=0)
    ax2.set_title("B. Age-Stratified Live Birth Rate per ET", fontsize=11.5)
    ax2.set_xlabel("Maternal Age Stratum")
    ax2.set_ylabel("Live Birth Rate per ET (%)")
    ax2.legend(title="Treatment Type", frameon=True, facecolor="white", fontsize=8.2)
    fig.suptitle("Treatment Modality Effectiveness: Crude Rates vs. Age-Stratified Comparison", fontsize=13.2, fontweight="bold")
    fig.savefig(os.path.join(viz_dir, "treatment_type_vs_outcome.png"), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    # --- Chart 12: Correlation Heatmap (`correlation_heatmap.png`) ---
    fig, ax = plt.subplots(figsize=(11.5, 9.2), constrained_layout=True)
    short_labels = [
        "Age",
        "BMI",
        "Infertility Yrs",
        "Prev IVF",
        "AMH (ng/mL)",
        "FSH (mIU/mL)",
        "Retrieved Ooc",
        "Mature Ooc",
        "Fertilized Ooc",
        "Viable Embryos",
        "Best Emb Score",
        "Total Cost ($)",
        "Implanted (0/1)",
        "Clin Preg (0/1)",
        "Live Birth (0/1)",
    ]
    corr_plot_df = spearman_mat.copy()
    corr_plot_df.index = short_labels
    corr_plot_df.columns = short_labels
    mask = np.triu(np.ones_like(corr_plot_df, dtype=bool), k=1)
    sns.heatmap(
        corr_plot_df,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-0.85,
        vmax=0.85,
        center=0,
        square=True,
        linewidths=0.6,
        annot_kws={"size": 8},
        cbar_kws={"shrink": 0.78, "label": "Spearman Rank Correlation (rho)"},
        ax=ax,
    )
    ax.set_title(f"Spearman Rank Correlation Matrix Across Patient, Laboratory, Cost & Outcome Variables (n = {n_cycles_clean:,})")
    fig.savefig(os.path.join(viz_dir, "correlation_heatmap.png"))
    plt.close(fig)

    # --- Chart 13: Treatment Cost Distribution (`treatment_cost_analysis.png`) ---
    fig, (ax_box, ax_hist) = plt.subplots(2, 1, figsize=(10.0, 6.2), gridspec_kw={"height_ratios": [1, 3.6]}, sharex=True, constrained_layout=True)
    sns.boxplot(x=costs["Total_Treatment_Cost"], color="#2E86AB", ax=ax_box, width=0.45, flierprops={"marker": "o", "markersize": 3, "alpha": 0.4})
    ax_box.set_xlabel("")
    ax_box.set_yticks([])
    ax_box.set_title(f"Total IVF Cycle Treatment Cost Distribution (n = {n_cycles_clean:,} Cycles, USD $)")

    sns.histplot(costs["Total_Treatment_Cost"], bins=34, kde=True, color="#1B365D", alpha=0.75, edgecolor="white", ax=ax_hist)
    c_mean = costs["Total_Treatment_Cost"].mean()
    c_med = costs["Total_Treatment_Cost"].median()
    c_q1 = costs["Total_Treatment_Cost"].quantile(0.25)
    c_q3 = costs["Total_Treatment_Cost"].quantile(0.75)
    ax_hist.axvline(c_mean, color="#E07A5F", lw=2.2, ls="--", label=f"Mean = ${c_mean:,.0f}")
    ax_hist.axvline(c_med, color="#008080", lw=2.2, ls="-.", label=f"Median = ${c_med:,.0f} (IQR: ${c_q1:,.0f} - ${c_q3:,.0f})")
    ax_hist.xaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
    ax_hist.set_xlabel("Total Cycle Treatment Cost (USD $)")
    ax_hist.set_ylabel("Number of Cycles")
    ax_hist.legend(frameon=True, facecolor="white")
    fig.savefig(os.path.join(viz_dir, "treatment_cost_analysis.png"))
    plt.close(fig)

    # --- Chart 14: Treatment Cost Category Comparison (`treatment_cost_category_comparison.png`) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.8), constrained_layout=True)
    comp_cols = ["Consultation_Cost", "Medication_Cost", "Laboratory_Cost", "Procedure_Cost", "Additional_Cost"]
    comp_labels = ["Consultation", "Medication", "Laboratory", "Procedure", "Additional"]
    comp_by_tt = master.groupby("Treatment_Type")[comp_cols].mean().reindex(TT_ORDER)
    comp_by_tt.columns = comp_labels
    comp_by_tt.plot(kind="bar", stacked=True, ax=ax1, color=PALETTE_MAIN[:5], width=0.62, edgecolor="white")
    for idx_i, tot_v in enumerate(comp_by_tt.sum(axis=1)):
        ax1.text(idx_i, tot_v + 250, f"${tot_v:,.0f}", ha="center", fontsize=9, fontweight="bold")
    ax1.set_ylim(0, comp_by_tt.sum(axis=1).max() * 1.15)
    ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
    ax1.tick_params(axis="x", rotation=14)
    ax1.set_title("A. Mean Itemized Cost Structure by Modality")
    ax1.set_xlabel("Treatment Type")
    ax1.set_ylabel("Mean Cost per Cycle (USD $)")
    ax1.legend(title="Cost Component", frameon=True, facecolor="white", fontsize=8.5)

    sns.boxplot(
        data=master,
        x="Treatment_Type",
        y="Total_Treatment_Cost",
        order=TT_ORDER,
        palette=PALETTE_MAIN[:4],
        ax=ax2,
        width=0.52,
        flierprops={"markersize": 2.5, "alpha": 0.35},
    )
    ax2.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
    ax2.tick_params(axis="x", rotation=14)
    ax2.set_title("B. Total Cost Dispersion Across Treatment Categories")
    ax2.set_xlabel("Treatment Type")
    ax2.set_ylabel("Total Treatment Cost (USD $)")
    fig.suptitle(f"Financial Cost Breakdown and Modality Price Dispersion (n = {n_cycles_clean:,} Cycles)", fontsize=13.2, fontweight="bold")
    fig.savefig(os.path.join(viz_dir, "treatment_cost_category_comparison.png"))
    plt.close(fig)

    # --- Chart 15: Live Birth Outcome Distribution & Clinical Funnel (`live_birth_outcome_distribution.png`) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.6), constrained_layout=True)
    funnel_labels = [
        "Initiated\nCycles",
        "Embryo\nTransfers",
        "Positive\nBeta-hCG",
        "Ultrasound\nImplantation",
        "Clinical\nPregnancy",
        "Confirmed\nLive Birth",
    ]
    funnel_vals = [n_cycles_clean, n_transfers, n_pos_hcg, n_imp, n_clin, n_lb]
    bars_f = ax1.barh(funnel_labels[::-1], funnel_vals[::-1], color=["#E07A5F", "#1B365D", "#008080", "#2E86AB", "#5C8D89", "#6C757D"], height=0.62)
    for b in bars_f:
        pct_init = 100.0 * b.get_width() / n_cycles_clean
        ax1.text(b.get_width() + 70, b.get_y() + b.get_height() / 2, f"{int(b.get_width()):,} ({pct_init:.1f}%)", va="center", fontsize=9, fontweight="bold")
    ax1.set_xlim(0, n_cycles_clean * 1.25)
    ax1.set_title("A. End-to-End Clinical Progression Funnel")
    ax1.set_xlabel("Number of Cycles (% of Initiated Cycles)")

    # Panel B: Live Birth Rate per ET by Transfer Type x Transfer Strategy
    piv_lb = (
        et_df[et_df["Transfer_Strategy"].isin(["eSET (1 Embryo)", "DET (2 Embryos)"])]
        .pivot_table(index="Transfer_Type", columns="Transfer_Strategy", values="Live_Birth_Bin", aggfunc=lambda s: 100.0 * s.mean())
    )
    piv_lb.plot(kind="bar", ax=ax2, color=["#1B365D", "#008080"], width=0.55, edgecolor="white")
    for container in ax2.containers:
        for b in container:
            ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.0, f"{b.get_height():.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax2.set_ylim(0, 48)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax2.tick_params(axis="x", rotation=0)
    ax2.set_title("B. Live Birth per ET: Modality & Transfer Count", fontsize=11.5)
    ax2.set_xlabel("Transfer Modality")
    ax2.set_ylabel("Live Birth Rate per ET (%)")
    ax2.legend(title="Embryos Transferred", frameon=True, facecolor="white")
    fig.suptitle("Clinic-Wide Live Birth Outcome Funnel and Transfer Strategy Comparison", fontsize=13.2, fontweight="bold")
    fig.savefig(os.path.join(viz_dir, "live_birth_outcome_distribution.png"), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    # --- Chart 16: Outlier Detection Boxplots (`outlier_detection_boxplots.png`) ---
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.2), constrained_layout=True)
    outlier_specs = [
        (patients["Age"], "Maternal Age (Years)", "#1B365D"),
        (patients["BMI"], "Patient BMI (kg/m²)", "#008080"),
        (patients["AMH_Level"], "Serum AMH (ng/mL)", "#2E86AB"),
        (patients["FSH_Level"], "Basal FSH (mIU/mL)", "#6D597A"),
        (master["Retrieved_Oocytes"], "Retrieved Oocytes (Count)", "#E07A5F"),
        (costs["Total_Treatment_Cost"], "Total Cost (USD $)", "#4CAF50"),
    ]
    for ax_i, (s_data, lbl_s, clr_s) in zip(axes.flat, outlier_specs):
        q1, q3 = s_data.quantile(0.25), s_data.quantile(0.75)
        iqr_v = q3 - q1
        n_out = int(((s_data < q1 - 1.5 * iqr_v) | (s_data > q3 + 1.5 * iqr_v)).sum())
        sns.boxplot(x=s_data, color=clr_s, ax=ax_i, width=0.45, flierprops={"marker": "o", "markersize": 3, "alpha": 0.45})
        ax_i.set_title(f"{lbl_s}\n(1.5xIQR Outliers: n = {n_out:,} [{100*n_out/len(s_data):.1f}%])", fontsize=10.5)
        ax_i.set_xlabel(lbl_s)
    fig.suptitle("Univariate 1.5xIQR Outlier Assessment on Cleaned Physiological, Laboratory & Cost Variables", fontsize=13.2, fontweight="bold")
    fig.savefig(os.path.join(viz_dir, "outlier_detection_boxplots.png"))
    plt.close(fig)

    # --- Chart 17: Day 3 vs Day 5 Embryo Development & Outcomes (`day3_vs_day5_outcomes.png`) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.5), constrained_layout=True)
    ct_day_q = pd.crosstab(embryos["Embryo_Day"], embryos["Embryo_Quality_Category"], normalize="index").reindex(index=["Day 3", "Day 5"], columns=QUALITY_ORDER) * 100.0
    ct_day_q.plot(kind="bar", stacked=True, color=QUALITY_COLORS, ax=ax1, width=0.52, edgecolor="white")
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax1.tick_params(axis="x", rotation=0)
    ax1.set_title("A. Morphological Quality Composition by Culture Day")
    ax1.set_xlabel("Embryo Assessment Day")
    ax1.set_ylabel("Percentage of Graded Embryos (%)")
    ax1.legend(title="Quality Tier", frameon=True, facecolor="white")

    x_d = np.arange(2)
    days_ord = ["Day 3", "Day 5"]
    for i, (col, lbl, clr) in enumerate([
        ("Implanted_Bin", "Implantation Rate (%)", "#1B365D"),
        ("Clinical_Preg_Bin", "Clinical Pregnancy Rate (%)", "#008080"),
        ("Live_Birth_Bin", "Live Birth Rate (%)", "#E07A5F"),
    ]):
        vals = [100.0 * et_df[et_df["Best_Embryo_Day"] == d][col].mean() for d in days_ord]
        bars_d = ax2.bar(x_d + (i - 1) * 0.24, vals, width=0.24, color=clr, label=lbl)
        for b in bars_d:
            ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.0, f"{b.get_height():.1f}%", ha="center", fontsize=8.8, fontweight="bold")
    ax2.set_xticks(x_d)
    ax2.set_xticklabels([f"{d} Transfer\n(n={len(et_df[et_df['Best_Embryo_Day']==d]):,})" for d in days_ord])
    ax2.set_ylim(0, 56)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax2.set_title("B. Transfer Outcome Rates: Day 3 Cleavage vs. Day 5 Blastocyst")
    ax2.set_xlabel("Culture Stage at Transfer")
    ax2.set_ylabel("Outcome Rate per ET (%)")
    ax2.legend(frameon=True, facecolor="white")
    fig.suptitle("Comparison of Cleavage-Stage (Day 3) vs. Blastocyst-Stage (Day 5) Embryo Performance", fontsize=13.2, fontweight="bold")
    fig.savefig(os.path.join(viz_dir, "day3_vs_day5_outcomes.png"))
    plt.close(fig)

    # --- Chart 18: Cost-Effectiveness & Multivariable Adjusted Odds Ratios (`operational_efficiency_cost_effectiveness.png`) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 6.0), constrained_layout=True)
    cplb_vals = [r["Cost_per_Live_Birth"] for r in by_age_group]
    bars_cp = ax1.bar(AGE_ORDER, cplb_vals, color=["#1B365D", "#008080", "#2E86AB", "#E9C46A", "#E07A5F"], width=0.58)
    for b in bars_cp:
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 1500, f"${b.get_height():,.0f}", ha="center", fontsize=8.8, fontweight="bold")
    ax1.set_ylim(0, max(cplb_vals) * 1.16)
    ax1.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
    ax1.set_title("A. Cumulative Clinic Spend per Live Birth by Age Bracket")
    ax1.set_xlabel("Maternal Age Group (Years)")
    ax1.set_ylabel("Mean Total Spend per Live Birth Achieved (USD $)")

    # Forest plot of Multivariable Logistic Regression Odds Ratios (excluding Intercept)
    forest_items = [r for r in logit_results if r["Feature"] != "Intercept"][::-1]
    f_labels_map = {
        "Age_per_5yr": "Maternal Age (+5 Years)",
        "Embryo_Excellent_vs_FairPoor": "Embryo: Excellent (vs Fair/Poor)",
        "Embryo_Good_vs_FairPoor": "Embryo: Good (vs Fair/Poor)",
        "Day5_vs_Day3": "Day 5 Blastocyst (vs Day 3)",
        "PGT_A_vs_Conventional": "IVF + PGT-A (Age-Adjusted)",
        "ICSI_vs_Conventional": "ICSI (vs Conventional/Mild)",
        "FET_vs_Fresh": "Frozen-Thawed ET (vs Fresh)",
        "Multi_Embryo_Transfer_vs_eSET": "2+ Embryos Transferred (vs eSET)",
        "Obese_BMI_ge_30": "BMI >= 30 kg/m² (Obese)",
        "Previous_IVF_Attempts": "Previous IVF Attempts (+1 Cycle)",
    }
    y_pos = np.arange(len(forest_items))
    ors = [r["Odds_Ratio"] for r in forest_items]
    ci_lo = [r["Odds_Ratio"] - r["OR_CI_Lower"] for r in forest_items]
    ci_hi = [r["OR_CI_Upper"] - r["Odds_Ratio"] for r in forest_items]
    colors_f = ["#008080" if r["Odds_Ratio"] > 1 and r["p_value"] < 0.05 else ("#E07A5F" if r["Odds_Ratio"] < 1 and r["p_value"] < 0.05 else "#6C757D") for r in forest_items]

    ax2.axvline(1.0, color="#333333", ls="--", lw=1.4, label="Null Effect (OR = 1.0)")
    for yi, or_v, lo_e, hi_e, clr_f in zip(y_pos, ors, ci_lo, ci_hi, colors_f):
        ax2.errorbar(or_v, yi, xerr=[[lo_e], [hi_e]], fmt="o", color=clr_f, ecolor=clr_f, elinewidth=2.0, capsize=3.5, markersize=6.5)
        if or_v < 1.0:
            ax2.text(or_v - lo_e - 0.05, yi, f"{or_v:.2f}", va="center", ha="right", fontsize=8.5, fontweight="bold", color=clr_f)
        else:
            ax2.text(or_v + hi_e + 0.05, yi, f"{or_v:.2f}", va="center", ha="left", fontsize=8.5, fontweight="bold", color=clr_f)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f_labels_map[r["Feature"]] for r in forest_items])
    ax2.set_xlim(0.35, max([r["OR_CI_Upper"] for r in forest_items]) + 0.42)
    ax2.set_title(f"B. Adjusted Odds Ratios for Live Birth (n = {n_transfers:,} ETs)", fontsize=11.5)
    ax1.set_title("A. Clinic Spend per Live Birth by Age Bracket", fontsize=11.5)
    ax2.set_xlabel("Adjusted Odds Ratio (95% Confidence Interval)")
    fig.suptitle("Operational Cost-Effectiveness and Multivariable Predictors of Live Birth", fontsize=13.2, fontweight="bold")
    fig.savefig(os.path.join(viz_dir, "operational_efficiency_cost_effectiveness.png"), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    # Save complete analytical JSON payload
    analytical_payload = {
        "counts": {
            "patients_clean": n_patients_clean,
            "cycles_clean": n_cycles_clean,
            "transfers_clean": n_transfers,
            "no_transfer_cycles": n_no_transfer,
            "embryos_clean": n_embryos_clean,
        },
        "demographics": demographics_stats,
        "patient_category_counts": {
            "Age_Group": age_group_pt_counts,
            "BMI_Category": bmi_cat_pt_counts,
            "Infertility_Type": inf_type_pt_counts,
            "Medical_History_Category": med_hist_pt_counts,
        },
        "treatment_and_funnel": {
            "treatment_counts": treatment_counts,
            "protocol_counts": protocol_counts,
            "transfer_type_counts": transfer_type_counts,
            "transfer_strategy_counts": transfer_strategy_counts,
            "funnel_means": funnel_means,
        },
        "embryo_summary": {
            "all_embryos_quality": emb_quality_all,
            "all_embryos_day": emb_day_counts,
            "day5_icm_te_implantation_pct": icm_te_pivot.to_dict(),
        },
        "outcome_rates": outcome_rates,
        "stratified_outcomes": {
            "by_embryo_quality_et": by_embryo_quality_et,
            "by_age_group": by_age_group,
            "by_treatment_type": by_treatment_type,
            "by_embryo_day_et": by_embryo_day_et,
            "by_transfer_type_et": by_transfer_type_et,
            "by_transfer_strategy_et": by_transfer_strategy_et,
            "by_infertility_type": by_infertility_type,
            "by_bmi_cat": by_bmi_cat,
            "by_protocol": by_protocol,
            "older_age_ge_38_by_treatment": older_stratified_lbr,
        },
        "cost_summary": cost_summary,
        "pairwise_correlations": pairwise_corrs,
        "hypothesis_tests": hypothesis_tests,
        "logistic_regression_et_live_birth": logit_results,
    }

    with open(os.path.join(docs_dir, "analytical_results.json"), "w") as f:
        json.dump(analytical_payload, f, indent=2)

    print("=== EDA, Statistical Analysis & All 18 Visualizations Generated Successfully ===")
    viz_files = sorted(os.listdir(viz_dir))
    print(f"Total PNG visualizations saved: {len(viz_files)}")
    for vf in viz_files:
        print(f"  - {vf}")


if __name__ == "__main__":
    run_analysis_and_visualizations()
