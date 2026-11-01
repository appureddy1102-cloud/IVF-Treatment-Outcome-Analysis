# Data Quality Assessment & Preprocessing Report

> **Synthetic Data Disclaimer:** All datasets evaluated in this report are 100% synthetic and generated programmatically for healthcare analytics evaluation. No real patient data is included.

## 1. Executive Summary of Cleaning Audit

| Raw Dataset | Original Rows | Missing Cells (Raw) | Duplicates Removed | Invalid / Orphan Removed | Imputed / Reconciled | Final Clean Rows | Residual Missing |
|---|---:|---:|---:|---:|---:|---:|---:|
| `patients.csv` | 3,925 | 281 | 25 | 3 | 7 | **3,897** | 0 |
| `ivf_treatment_cycles.csv` | 5,365 | 34 | 45 | 43 | 194 | **5,277** | 0 |
| `embryo_grading.csv` | 17,072 | 266 | 35 | 109 | 262 | **16,928** | 0 |
| `implantation_outcomes.csv` | 5,340 | 42 | 20 | 43 | 42 | **5,277** | 0 |
| `treatment_costs.csv` | 5,338 | 156 | 18 | 43 | 332 | **5,277** | 0 |

## 2. Detailed Cleaning Decisions & Justifications

### `patients.csv` -> `cleaned_patients.csv`
- **Audit Summary:** Removed 25 exact duplicate Patient_ID rows and 3 records with impossible maternal ages (-29, 14, 71). Converted 7 impossible BMI/AMH/FSH entries to NaN and applied diagnosis/age-stratified median imputation. Standardized casing/whitespace in Infertility_Type and Medical_History_Category.

### `ivf_treatment_cycles.csv` -> `cleaned_treatment_cycles.csv`
- **Audit Summary:** Removed 45 duplicate Cycle_ID rows, 11 orphan/invalid-patient FK cycles, 18 biological funnel violations (e.g., Mature > Retrieved or Transferred > Available), and 14 cycles with contradictory clinical outcome cascades. Standardized 160 non-ISO dates to YYYY-MM-DD and imputed 34 missing Stimulation_Protocol values.

### `embryo_grading.csv` -> `cleaned_embryo_grading.csv`
- **Audit Summary:** Removed 35 duplicate Embryo_ID rows and 109 orphan/excluded-cycle embryo records (5 synthetic orphan Cycle_IDs + embryos belonging to excluded invalid cycles). Deterministically recovered 138 missing ICM grades and 124 missing TE grades on Day 5 blastocysts directly from composite Morphological_Grade strings.

### `implantation_outcomes.csv` -> `cleaned_implantation_outcomes.csv`
- **Audit Summary:** Removed 20 duplicate Outcome_ID rows, excluded 14 contradictory clinical cascade records (e.g., Live Birth='Yes' with Clinical Pregnancy='No'), aligned 1:1 with valid cycles, standardized status strings, and resolved 42 missing Follow_Up_Status entries.

### `treatment_costs.csv` -> `cleaned_treatment_costs.csv`
- **Audit Summary:** Removed 18 duplicate Cost_ID rows, aligned 1:1 with valid cycles, corrected 4 negative sign-entry errors in Consultation_Cost, imputed 44 missing Medication_Cost and 111 missing Additional_Cost values via protocol/modality medians, and reconciled 173 Total_Treatment_Cost sum discrepancies.

## 3. Remaining Data-Quality Limitations

1. **Median Imputation Attenuation:** Group-stratified median imputation for missing `BMI`, `AMH_Level`, `FSH_Level`, `Medication_Cost`, and `Additional_Cost` (~0.8%–2.2% of records) preserves subgroup central tendency without dropping valid clinical cycles, slightly attenuating extreme tail variance in those variables.
2. **Right-Censoring in Ongoing Follow-Up:** A small fraction of clinical pregnancies with `Follow_Up_Status = 'Ongoing Follow-Up'` or `'Lost to Follow-Up'` do not have confirmed live birth delivery recorded; primary live birth proportions use conservative intention-to-treat / per-transfer denominators and sensitivity checks confirm negligible bias.
3. **Retained Biological Outliers:** High-AMH / high-oocyte PCOS responses and elevated BMI values flagged by univariate `1.5 * IQR` fences were intentionally retained after range verification because they represent clinically authentic patient subpopulations rather than measurement errors.
