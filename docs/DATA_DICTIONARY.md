# Data Dictionary

All data are synthetic, generated with NumPy seed 42. No real patient records are included. Definitions describe the raw fields; cleaned tables include engineered features as noted. Units are descriptive and generated values are not clinical reference ranges.

## patients.csv
| Variable | Type | Description / expected values | Analytical purpose |
|---|---|---|---|
| Patient_ID | string | Unique synthetic patient key | Primary key / joins |
| Age | integer | Age in years; cleaned valid range 18–50 | Demographic stratification |
| BMI | float | Body mass index, kg/m² | Descriptive subgroup analysis |
| Infertility_Duration_Years | float | Duration in years | Patient history |
| Infertility_Type | category | Primary / Secondary | Cohort comparisons |
| Previous_IVF_Attempts | integer | Nonnegative prior attempt count | Treatment history |
| AMH_Level | float | Synthetic ovarian-reserve marker, ng/mL | Descriptive association |
| FSH_Level | float | Synthetic hormone marker, mIU/mL | Descriptive association |
| Medical_History_Category | category | Male Factor, Unexplained Infertility, Ovulatory Dysfunction / PCOS, Tubal Factor, Diminished Ovarian Reserve, Endometriosis, Uterine Factor | Cohort description |

## ivf_treatment_cycles.csv
| Variable | Type | Description / expected values | Analytical purpose |
|---|---|---|---|
| Cycle_ID | string | Unique synthetic cycle key | Primary key / joins |
| Patient_ID | string | Patient foreign key | Link to patient |
| Treatment_Date | date | ISO date after cleaning | Time description |
| Treatment_Type | category | ICSI, Conventional IVF, IVF + PGT-A, Mild Stimulation IVF | Modality comparison |
| Stimulation_Protocol | category | GnRH Antagonist, GnRH Agonist Long, Mild / Minimal Stimulation, Flare-Up Protocol | Protocol comparison |
| Retrieved_Oocytes | integer | Retrieved oocyte count | Laboratory funnel |
| Mature_Oocytes | integer | Mature count; <= retrieved | Funnel validation |
| Fertilized_Oocytes | integer | Fertilized count; <= mature | Funnel validation |
| Fertilization_Method | category | ICSI / Conventional IVF | Laboratory methods |
| Number_of_Embryos | integer | Viable/developed embryo count | Development funnel |
| Number_of_Embryos_Transferred | integer | Count transferred; <= embryos | Transfer analysis |
| Transfer_Type | category | Fresh Transfer / Frozen-Thawed Transfer / No Transfer | Transfer stratification |
| Engineered fields | mixed | Age_Group, BMI_Category, oocyte/embryo rates, transfer features, and binary outcome features where present in cleaned output | Analysis and reporting |

## embryo_grading.csv
| Variable | Type | Description / expected values | Analytical purpose |
|---|---|---|---|
| Embryo_ID | string | Unique synthetic embryo key | Primary key |
| Cycle_ID | string | Cycle foreign key | Link to cycle/outcome |
| Embryo_Day | category | Day 3 / Day 5 | Developmental timing |
| Embryo_Stage | category | Cleavage / Blastocyst | Stage analysis |
| Expansion_Grade | category/integer | Gardner-like score 1–6 for Day 5; descriptive synthetic grading | Morphology description |
| Inner_Cell_Mass_Grade | category | A / B / C | Day 5 morphology |
| Trophectoderm_Grade | category | A / B / C | Day 5 morphology |
| Embryo_Quality_Category | category | Excellent / Good / Fair / Poor | Quality stratification |
| Morphological_Grade | category | Synthetic composite morphology label | Quality summary |
| Engineered fields | mixed | Numeric quality score / cycle-level lead embryo mapping where applicable | Analysis joins |

## implantation_outcomes.csv
| Variable | Type | Description / expected values | Analytical purpose |
|---|---|---|---|
| Outcome_ID | string | Unique synthetic outcome key | Record identification |
| Cycle_ID | string | Cycle foreign key, one outcome per cleaned cycle | Referential integrity |
| Implantation_Status | category | Yes / No | Implantation proportion |
| Pregnancy_Test_Result | category | Positive / Negative | Biochemical test endpoint |
| Clinical_Pregnancy_Status | category | Yes / No | Clinical pregnancy endpoint |
| Live_Birth_Status | category | Yes / No | Live-birth endpoint |
| Follow_Up_Status | category | Completed / Incomplete | Follow-up sensitivity |

## treatment_costs.csv
| Variable | Type | Description / expected values | Analytical purpose |
|---|---|---|---|
| Cost_ID | string | Unique synthetic cost record key | Record identification |
| Cycle_ID | string | Cycle foreign key | Join to treatment/outcome |
| Consultation_Cost | float | Synthetic monetary units (USD-labelled for charts) | Cost component |
| Medication_Cost | float | Synthetic monetary units | Cost component |
| Laboratory_Cost | float | Synthetic monetary units | Cost component |
| Procedure_Cost | float | Synthetic monetary units | Cost component |
| Additional_Cost | float | Synthetic monetary units | Cost component |
| Total_Treatment_Cost | float | Sum of components after cleaning | Cost outcome |
| Engineered fields | mixed | Component shares, cost per observed outcome summaries | Operational description |

## data_dictionary.csv
| Variable | Type | Description / expected values | Analytical purpose |
|---|---|---|---|
| variable | string | Field name | Lookup |
| data_type | string | Expected storage type | Import guidance |
| description | string | Plain-language field definition | Documentation |
| expected_values | string | Allowed values/range | Validation |
| analytical_purpose | string | Use in analysis | Governance |

## Denominator definitions
- **Initiated-cycle rate:** number of cycles reaching the named endpoint divided by all cleaned initiated cycles.
- **Per-transfer rate:** endpoint count divided by cycles with an embryo transfer; no-transfer cycles excluded from that denominator.
- **Implantation, clinical pregnancy, and live birth** are separate endpoints and not interchangeable.

All dollar labels are synthetic units and not a real clinic price list. This project is educational, contains no real patient records, and cannot guide clinical decisions.