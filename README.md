# IVF Treatment Outcome Analysis & Success Rate Optimization Using Python

> **Synthetic-data educational project. No actual patient records are used. Results must not be used to guide clinical patient decisions.**

## Project overview
A reproducible analytics project for a fictional fertility clinic, exploring patient and cycle characteristics, embryo morphology, IVF outcome endpoints, and treatment-cost structure. Six CSVs are generated with NumPy/Pandas using fixed seed 42; controlled data issues demonstrate transparent data cleaning. There are 5,000+ raw cycle records and 18 high-resolution charts.

## Objectives
- Profile and validate linked records and document each cleaning action.
- Describe patient demographics, laboratory funnel, embryo quality, outcomes, and costs.
- Distinguish initiated-cycle rates from per-embryo-transfer rates; report implantation, clinical pregnancy, and live birth separately.
- Use descriptive correlations and statistical tests with effect sizes, and avoid causal interpretation.
- Present synthetic operational monitoring ideas with explicit limitations.

## Project structure
```text
README.md / requirements.txt
src/                         # generation, cleaning, analysis, deliverable builder
 data/raw/                    # six synthetic source CSV files
 data/cleaned/                # cleaned CSVs
 notebooks/                   # executed analysis notebook
 reports/                     # Word report and internship description
 visualizations/              # 18 300-DPI PNG charts
 docs/                         # quality summary, data dictionary, audit
```

## Dataset
Raw data tables: patients, IVF treatment cycles, embryo grading, implantation outcomes, treatment costs, and data dictionary. The generator intentionally injects controlled missingness, duplicates, casing inconsistencies, foreign-key orphans and logical anomalies. The cleaning scripts explain valid removals and repairs. Clean outputs have zero missing values and validated keys/constraints. See [`docs/DATA_QUALITY_REPORT.md`](docs/DATA_QUALITY_REPORT.md), [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md), and [`docs/analytical_results.json`](docs/analytical_results.json).

## Reproducibility
Use Python 3.10+ (tested in Python 3.13). From repository root:
```bash
python -m pip install -r requirements.txt
python src/generate_synthetic_data.py
python src/clean_and_preprocess.py
python src/statistical_analysis_and_charts.py
jupyter nbconvert --to notebook --execute --inplace notebooks/IVF_Treatment_Outcome_Analysis.ipynb
```
The analysis operates on the cleaned tables and persists analytical results and charts. Notebook images are embedded inline when executed.

## Key results (generated data only)
- Cleaned cohort: 5,277 cycles, 4,672 embryo-transfer cycles, 16,928 embryos.
- Per initiated cycle: clinical pregnancy 38.30%, live birth 32.42%; per transfer: 43.26% and 36.62%, respectively.
- Lead transferred embryo quality groups show differing descriptive live-birth proportions (Excellent 42.04%, Poor 16.85% per transfer); not a causal estimate.
- Age-stratified transfer live-birth proportion ranges from 44.11% (<30) to 18.99% (>40) in this synthetic cohort.
- Mean total synthetic treatment cost: $13,439.89; median $13,446.29. Laboratory (32.48%) and medication (27.46%) are the largest itemized mean-spend shares.
- BMI's correlation with live birth is near zero and not statistically significant here (Spearman rho 0.010, p=0.491).

All figures are descriptive computations from the included generated dataset; they do not represent real clinical outcomes or treatment effects. Rates depend on denominators and must not be compared without matching definitions.

## Tools
Python, NumPy, pandas, SciPy, Matplotlib, Seaborn, scikit-learn, Jupyter, python-docx.

## Ethics and limitations
The project contains no personal/real patient data. Synthetic assumptions, repeated cycles, confounding and limited follow-up mean results are not generalizable. Correlation is not causation. This is not medical advice or a clinical decision tool.
