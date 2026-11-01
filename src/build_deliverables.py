from pathlib import Path
import json, os, textwrap
import pandas as pd
import nbformat as nbf
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parents[1]
(Root:=ROOT/'notebooks').mkdir(exist_ok=True)
R=ROOT/'reports'; R.mkdir(exist_ok=True)
d=json.load(open(ROOT/'docs/analytical_results.json'))
# Notebook: explanatory prose and reproducible end-to-end loading/analysis. Section headings are intentionally distinct.
sections=[
('1. Project Introduction','A reproducible synthetic-data analysis of IVF cycle patterns, outcomes and costs.'),
('2. Business Problem','A fictional clinic seeks operational indicators, outcome denominators, and cost patterns. No clinical decisions are supported.'),
('3. Objectives','Explore demographics, treatment and embryo features; quantify outcomes; test associations; communicate bounded operational insights.'),
('4. Dataset Source and Synthetic Data Disclaimer','All records are generated with NumPy seed 42. No real patient data or external patient records are used.'),
('5. Library Imports','Use pandas, numpy, scipy, matplotlib, seaborn, pathlib, JSON, and project scripts.'),
('6. Dataset Generation','Run the deterministic synthetic generator from the repository root when regeneration is required.'),
('7. Data Loading','Load the five cleaned tables and analytical results; raw sources remain available in data/raw.'),
('8. Data Understanding','Inspect shape, columns, dtypes, samples and categorical levels.'),
('9. Data Profiling','Review the stored profiling summary and before/after data quality report.'),
('10. Data Cleaning','Cleaning decisions are implemented in src/clean_and_preprocess.py and documented in docs/DATA_QUALITY_REPORT.md.'),
('11. Data Validation','Check keys, referential integrity, missing values, biological funnel constraints and exact cost totals.'),
('12. Feature Engineering','Cleaned cycle data includes derived outcome/rate and patient grouping features; analytical tables join patient, cycle, outcome and cost records.'),
('13. Exploratory Data Analysis','Summarize patient characteristics, cycle modalities, embryo quality, outcomes and itemized treatment costs.'),
('14. Statistical Analysis','Pearson/Spearman correlations, chi-square tests with Cramer’s V, rank tests with epsilon-squared, and multivariable logistic regression are reported descriptively.'),
('15. Data Visualization','Display all PNG charts inline from the visualizations folder.'),
('16. Key Findings','Outcome percentages below use explicitly stated denominators; estimates describe only these synthetic records.'),
('17. Business Recommendations','Monitor funnel completion, costs and outcomes with stratification and uncertainty; do not infer treatment effects.'),
('18. Limitations','Synthetic assumptions, repeated patients/cycles, potential residual confounding, follow-up, missingness repair, and no external validation limit inference.'),
('19. Conclusion','This reproducible workflow illustrates data quality, descriptive analytics and responsible reporting—not clinical evidence.')]
nb=nbf.v4.new_notebook(); cells=[]
cells.append(nbf.v4.new_markdown_cell('# IVF Treatment Outcome Analysis & Success Rate Optimization Using Python\n\n**Synthetic educational analytics project — not clinical evidence.**'))
for title,desc in sections:
    cells.append(nbf.v4.new_markdown_cell(f'## {title}\n\n{desc}'))
    if title=='5. Library Imports':
        cells.append(nbf.v4.new_code_cell('from pathlib import Path\nimport json\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nfrom IPython.display import display, Image\nROOT = Path.cwd()\nif not (ROOT / "data").exists(): ROOT = ROOT.parent\nprint("Project root:", ROOT)'))
    elif title=='6. Dataset Generation':
        cells.append(nbf.v4.new_code_cell('# Optional deterministic regeneration. Existing CSVs are used for the remaining analysis.\n# %run src/generate_synthetic_data.py\nprint("Generator: src/generate_synthetic_data.py (fixed seed = 42)")'))
    elif title=='7. Data Loading':
        cells.append(nbf.v4.new_code_cell('raw_dir=ROOT/"data/raw"; clean_dir=ROOT/"data/cleaned"\npatients=pd.read_csv(clean_dir/"cleaned_patients.csv")\ncycles=pd.read_csv(clean_dir/"cleaned_treatment_cycles.csv", parse_dates=["Treatment_Date"])\nembryos=pd.read_csv(clean_dir/"cleaned_embryo_grading.csv")\noutcomes=pd.read_csv(clean_dir/"cleaned_implantation_outcomes.csv")\ncosts=pd.read_csv(clean_dir/"cleaned_treatment_costs.csv")\nresults=json.load(open(ROOT/"docs/analytical_results.json"))\nprint({"patients":patients.shape,"cycles":cycles.shape,"embryos":embryos.shape,"outcomes":outcomes.shape,"costs":costs.shape})'))
    elif title=='8. Data Understanding':
        cells.append(nbf.v4.new_code_cell('display(cycles.head(10)); display(cycles.tail(10)); display(patients.describe(include="all").T.head(12))\nprint("Cycle dtypes:"); display(cycles.dtypes)'))
    elif title=='9. Data Profiling':
        cells.append(nbf.v4.new_code_cell('quality=json.load(open(ROOT/"docs/data_quality_summary.json")); print(json.dumps(quality,indent=2)[:10000])'))
    elif title=='10. Data Cleaning':
        cells.append(nbf.v4.new_code_cell('display(pd.read_csv(ROOT/"docs/iqr_outlier_summary.csv").head(20))\nprint((ROOT/"docs/DATA_QUALITY_REPORT.md").read_text()[:5000])'))
    elif title=='11. Data Validation':
        cells.append(nbf.v4.new_code_cell('assert not patients.Patient_ID.duplicated().any()\nassert not cycles.Cycle_ID.duplicated().any()\nassert not outcomes.Cycle_ID.duplicated().any()\nassert not costs.Cycle_ID.duplicated().any()\nassert cycles.Patient_ID.isin(patients.Patient_ID).all()\nassert outcomes.Cycle_ID.isin(cycles.Cycle_ID).all() and costs.Cycle_ID.isin(cycles.Cycle_ID).all()\nassert cycles.Mature_Oocytes.le(cycles.Retrieved_Oocytes).all()\nassert cycles.Number_of_Embryos_Transferred.le(cycles.Number_of_Embryos).all()\nassert (costs.Total_Treatment_Cost.round(2)==costs[["Consultation_Cost","Medication_Cost","Laboratory_Cost","Procedure_Cost","Additional_Cost"]].sum(axis=1).round(2)).all()\nprint("Validation passed; cleaned tables missing values:", int(sum(x.isna().sum().sum() for x in [patients,cycles,embryos,outcomes,costs])))'))
    elif title=='12. Feature Engineering':
        cells.append(nbf.v4.new_code_cell('print("Derived fields:", [c for c in cycles.columns if c not in pd.read_csv(clean_dir/"cleaned_treatment_cycles.csv").columns[:12]])\ndisplay(cycles.head())'))
    elif title=='13. Exploratory Data Analysis':
        cells.append(nbf.v4.new_code_cell('print("Outcome rates (counts and explicit denominators):")\nprint(json.dumps(results["outcome_rates"],indent=2))\nprint("\\nCost summary:")\nprint(json.dumps(results["cost_summary"],indent=2))'))
    elif title=='14. Statistical Analysis':
        cells.append(nbf.v4.new_code_cell('display(pd.DataFrame(results["hypothesis_tests"]))\ndisplay(pd.DataFrame(results["pairwise_correlations"]).head(20))\ndisplay(pd.DataFrame(results["logistic_regression_et_live_birth"]))'))
    elif title=='15. Data Visualization':
        cells.append(nbf.v4.new_code_cell('for f in sorted((ROOT/"visualizations").glob("*.png")):\n    print(f.name); display(Image(filename=str(f),width=800))'))
    elif title=='16. Key Findings':
        cells.append(nbf.v4.new_code_cell('print("Outcome overview:",results["outcome_rates"]["counts"])\nprint("Per initiated-cycle rates:",results["outcome_rates"]["per_initiated_cycle_pct"])\nprint("Per-transfer rates:",results["outcome_rates"]["per_embryo_transfer_pct"])\nprint("Embryo quality stratification:"); display(pd.DataFrame(results["stratified_outcomes"]["by_embryo_quality_et"]))\nprint("Age stratification:"); display(pd.DataFrame(results["stratified_outcomes"]["by_age_group"]))'))
    elif title=='17. Business Recommendations':
        cells.append(nbf.v4.new_markdown_cell('- Maintain stage-specific denominators and follow-up-completion reporting.\n- Monitor laboratory funnel conversion, costs, and stratified outcome variation.\n- Investigate operational variation in prospectively governed real-world data; the synthetic comparisons are not causal or prescriptive.'))
    elif title in ('18. Limitations','19. Conclusion'):
        cells.append(nbf.v4.new_markdown_cell('**Disclaimer:** Entirely synthetic data. Findings are demonstrative and cannot guide clinical patient decisions.'))
nb.cells=cells; nb.metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3"}}
nbf.write(nb,ROOT/'notebooks/IVF_Treatment_Outcome_Analysis.ipynb')

# Professional report, using real calculated JSON values and all generated charts.
doc=Document(); sec=doc.sections[0]; sec.top_margin=Inches(.72); sec.bottom_margin=Inches(.68); sec.left_margin=Inches(.78); sec.right_margin=Inches(.78)
styles=doc.styles; styles['Normal'].font.name='Calibri'; styles['Normal'].font.size=Pt(10); styles['Normal'].paragraph_format.space_after=Pt(5)
for nm,size,color in [('Title',28,'17365D'),('Heading 1',18,'17365D'),('Heading 2',13,'007C83')]:
 st=styles[nm]; st.font.name='Calibri'; st.font.size=Pt(size); st.font.color.rgb=RGBColor.from_string(color)
# page number footer
foot=sec.footer.paragraphs[0]; foot.alignment=WD_ALIGN_PARAGRAPH.CENTER; foot.add_run('Synthetic educational analysis  •  Page ')
fld=OxmlElement('w:fldSimple'); fld.set(qn('w:instr'),'PAGE'); foot._p.append(fld)
# cover
t=doc.add_paragraph(); t.alignment=WD_ALIGN_PARAGRAPH.CENTER; t.paragraph_format.space_before=Pt(95); r=t.add_run('IVF TREATMENT OUTCOME ANALYSIS\n& SUCCESS RATE OPTIMIZATION'); r.bold=True; r.font.size=Pt(25); r.font.color.rgb=RGBColor(23,54,93)
p=doc.add_paragraph('Using Python • Synthetic data analytics project'); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
p=doc.add_paragraph('\n\nInternship Assessment Submission\n\nPrepared: 1 November 2026\n\n'); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
p=doc.add_paragraph('IMPORTANT DISCLAIMER\nAll records and outcomes are fully synthetic. No actual patient data are used. This report is an educational analytics demonstration and must not be used to guide clinical decisions.'); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
doc.add_page_break()
doc.add_heading('Contents',0)
for s in ['Executive Summary','Introduction and Business Background','Problem Statement and Objectives','Dataset Description and Generation','Data Dictionary Summary','Data Understanding and Quality Assessment','Cleaning and Feature Engineering','Exploratory Data Analysis','Statistical Analysis','Visualization Gallery','Evidence-Based Findings','Business Recommendations','Limitations and Ethical Considerations','Conclusion','Tools, Reproducibility and Disclaimer']:
 doc.add_paragraph(s,style='List Bullet')
doc.add_page_break()
def para(s): doc.add_paragraph(s)
def table(headers,rows):
 tb=doc.add_table(rows=1,cols=len(headers)); tb.style='Light Shading Accent 1'
 for c,v in zip(tb.rows[0].cells,headers): c.text=str(v)
 for row in rows:
  cells=tb.add_row().cells
  for c,v in zip(cells,row): c.text=str(v)
def chart(name,caption,width=6.4):
 p=ROOT/'visualizations'/name
 if p.exists():
  q=doc.add_paragraph(); q.alignment=WD_ALIGN_PARAGRAPH.CENTER; q.add_run().add_picture(str(p),width=Inches(width))
  c=doc.add_paragraph(caption); c.alignment=WD_ALIGN_PARAGRAPH.CENTER
  if c.runs: c.runs[0].italic=True; c.runs[0].font.size=Pt(9)
# Sections with robust analytical content
sections_report=[
('1. Executive Summary',f"This project builds a reproducible synthetic-data pipeline and examines {d['counts']['cycles_clean']:,} cleaned treatment cycles, {d['counts']['transfers_clean']:,} embryo transfers and {d['counts']['embryos_clean']:,} embryo records. Across initiated cycles, clinical pregnancy was {d['outcome_rates']['per_initiated_cycle_pct']['Clinical_Pregnancy_Rate']:.2f}% and live birth was {d['outcome_rates']['per_initiated_cycle_pct']['Live_Birth_Rate']:.2f}%; among embryo transfers, the corresponding rates were {d['outcome_rates']['per_embryo_transfer_pct']['Clinical_Pregnancy_Rate']:.2f}% and {d['outcome_rates']['per_embryo_transfer_pct']['Live_Birth_Rate']:.2f}%. These are calculated rates in generated data only. Embryo quality and age groups showed descriptive gradients; treatment differences require cautious, non-causal interpretation."),
('2. Introduction and Business Background','A fictional fertility clinic seeks descriptive performance indicators spanning patient cohorts, laboratory development, embryo morphology, outcomes, and costs. The analysis models a complete data lifecycle—from seeded data generation to transparent cleaning, testing and communication.'),
('3. Problem Statement and Objectives','The practical question is how a clinic might organize operational monitoring while avoiding causal claims. Objectives: validate linked data; profile quality; describe treatment and outcome distributions; distinguish initiated-cycle and transfer denominators; test selected associations; summarize costs; and propose responsible monitoring.'),
('4. Dataset Description and Generation',f"Six raw CSVs were generated using NumPy/Pandas and fixed seed 42. The raw cycle table contains 5,365 rows and includes 5,000+ synthetic cycle records before cleaning. Linked tables describe {len(pd.read_csv(ROOT/'data/raw/patients.csv')):,} patients, cycles, embryo morphology, outcomes, itemized costs and a data dictionary. Controlled missingness, duplicate rows, casing variations and deliberate anomalies were introduced solely to demonstrate cleaning. Valid records enforce biological count ordering and cost-sum consistency."),
('5. Data Dictionary Summary','Patient variables capture age, BMI, ovarian markers and history; cycle variables record modality, stimulation, laboratory funnel counts and transfer; embryo variables hold developmental day, expansion and ICM/TE grades; outcome variables record sequential pregnancy endpoints and follow-up; cost variables itemize components. Full field-level definitions and expected values are provided in docs/DATA_DICTIONARY.md.'),
('6. Data Understanding and Quality Assessment','Initial profiling includes first/last records, shapes, types, summary statistics, missingness, duplicates, unique categories, primary/foreign keys and logical checks. Profiling evidence is preserved in docs/data_quality_summary.json and docs/DATA_QUALITY_REPORT.md. The cohort is synthetic and represents neither a population nor a clinic.'),
('7. Cleaning and Feature Engineering','Cleaning removes documented duplicate keys/records, invalid-age patients and dependent cycles, orphan foreign keys, contradictory biological funnels and invalid outcome cascades. Category labels and dates are standardized; missing values are imputed by documented context where appropriate; 173 cost-total discrepancies were reconciled to component sums. IQR flags identify distributional extremes but are retained unless logically invalid. Full removals and counts are itemized in the quality report.'),
('8. Exploratory Data Analysis',f"The cleaned cohort has mean age {d['demographics']['Age']['mean']:.2f} years (SD {d['demographics']['Age']['sd']:.2f}), mean BMI {d['demographics']['BMI']['mean']:.2f}, and mean total cost ${d['cost_summary']['Total_Treatment_Cost']['mean']:,.2f} (median ${d['cost_summary']['Total_Treatment_Cost']['median']:,.2f}). Of all initiated cycles, {d['outcome_rates']['counts']['Embryo_Transfer_Cycles']:,} proceeded to transfer. Outcome funnel and subgroup analyses are descriptive."),
('9. Statistical Analysis','Binary outcomes were compared with Pearson chi-square tests, reporting Cramer’s V and expected-cell checks. Continuous skewed/counted outcomes use Kruskal–Wallis rank tests with epsilon-squared; Welch comparisons are complemented by nonparametric checks. Pearson and Spearman coefficients are both retained where relevant. The multivariable logistic model reports odds ratios and confidence intervals for selected covariates among transfers. Observations may include repeat cycles per patient, violating strict independence; residual confounding, model specification and synthetic data-generating assumptions limit inferential interpretation.'),
('10. Visualizations','Eighteen 300-DPI charts cover demographics, lab funnel, morphology, outcome denominators, cost distribution, outliers, correlations, treatment stratification and model estimates. Captions describe the analytical purpose; charts are generated by src/statistical_analysis_and_charts.py.')]
for title,body in sections_report:
 doc.add_heading(title,1); para(body)
 if title.startswith('4.'):
  table(['Table','Raw dimensions','Role'],[['patients.csv','3,925 × 9','Patient characteristics'],['ivf_treatment_cycles.csv','5,365 × 12','Cycle and laboratory funnel'],['embryo_grading.csv','17,072 × 9','Embryo morphology'],['implantation_outcomes.csv','5,340 × 7','Follow-up outcomes'],['treatment_costs.csv','5,338 × 8','Cost components'],['data_dictionary.csv','45 × 6','Variable definitions']])
 if title.startswith('6.'):
  table(['Dataset','Before','After','Key documented actions'],[['Patients','3,925','3,897','25 duplicates; 3 invalid ages; range corrections and imputation'],['Cycles','5,365','5,277','45 duplicates; FK, funnel and contradictory-cascade exclusions'],['Embryos','17,072','16,928','35 duplicates; orphan cycles; recovered missing grades'],['Outcomes','5,340','5,277','20 duplicates; invalid cycles; follow-up resolution'],['Costs','5,338','5,277','18 duplicates; component imputation and total reconciliation']])
 if title.startswith('8.'):
  table(['Cost component','Mean USD','Share of total spend'],[[k.replace('_Cost',''),f"${v['mean']:,.2f}",f"{v['share_of_total_spend_pct']:.2f}%"] for k,v in d['cost_summary'].items() if isinstance(v,dict) and k!='Total_Treatment_Cost'])
 if title.startswith('9.'):
  table(['Test','N','Statistic','p value','Effect size'],[[x['Test_Name'],x['Sample_N'],x['Statistic'],f"{x['p_value']:.3g}",f"{x['Effect_Size_Metric']}: {x['Effect_Size']}"] for x in d['hypothesis_tests']])
 if title.startswith('10.'):
  chart('age_distribution.png','Figure 1. Synthetic patient age distribution.')
  chart('embryo_quality.png','Figure 2. Embryo quality categories.')
  chart('implantation_analysis.png','Figure 3. Cycle outcomes by best transferred embryo quality; denominator is transfers.')
  doc.add_page_break(); chart('age_group_vs_clinical_pregnancy.png','Figure 4. Sequential outcomes by age; denominator is embryo transfers.')
  chart('treatment_cost_analysis.png','Figure 5. Synthetic treatment-cost distribution.')
  doc.add_page_break(); chart('treatment_cost_category_comparison.png','Figure 6. Itemized cost comparison by modality.')
  chart('correlation_heatmap.png','Figure 7. Spearman correlation matrix; association is not causation.')
  doc.add_page_break(); chart('treatment_type_vs_outcome.png','Figure 8. Crude and age-stratified modality comparison; not causal.')
  chart('operational_efficiency_cost_effectiveness.png','Figure 9. Cost-per-observed-live-birth and adjusted model associations; not decision guidance.')
  doc.add_page_break(); chart('embryo_grading_comparison.png','Figure 10. Day 5 morphology and implantation mapping.')
  chart('live_birth_outcome_distribution.png','Figure 11. Outcome funnel and transfer strategy descriptions.')
  chart('outlier_detection_boxplots.png','Figure 12. IQR outlier audit; statistical flags are not necessarily errors.')
# Detailed findings, recommendations, ethics, remaining visualizations
for title,body in [
('11. Evidence-Based Findings',f"(1) Lead-embryo quality gradient: excellent-quality transfer cycles show {d['stratified_outcomes']['by_embryo_quality_et'][0]['LBR_per_ET_Pct']:.2f}% live birth per transfer versus {d['stratified_outcomes']['by_embryo_quality_et'][-1]['LBR_per_ET_Pct']:.2f}% for poor-quality transfers; this is an association in synthetic data, with subgroup selection and uncertainty. (2) Age gradient: transfer live-birth proportion is {d['stratified_outcomes']['by_age_group'][0]['LBR_per_ET_Pct']:.2f}% below 30 and {d['stratified_outcomes']['by_age_group'][-1]['LBR_per_ET_Pct']:.2f}% above 40. (3) Day 5 versus Day 3 best-transfer groups show {d['stratified_outcomes']['by_embryo_day_et'][0]['LBR_per_ET_Pct']:.2f}% versus {d['stratified_outcomes']['by_embryo_day_et'][1]['LBR_per_ET_Pct']:.2f}% live birth per transfer. (4) BMI and live birth association is weak and nonsignificant in this generated cohort (Spearman rho 0.010, p=0.491). (5) IVF+PGT-A has higher crude cost and outcome rate than conventional IVF, but these unrandomized simulated groups do not estimate a treatment effect."),
('12. Business Recommendations','Use a transparent cycle funnel dashboard, reporting initiated-cycle and transfer denominators separately, follow-up completion, subgroup counts and uncertainty intervals. Monitor missingness, cancellations, laboratory conversion, itemized cost shares and distributions over time. For real operational evaluation, preregister a protocol, use appropriately governed real-world data, account for repeat patients and selection, and involve clinical/statistical review. Do not use these findings to select treatment.'),
('13. Limitations and Ethical Considerations','This is entirely synthetic data and its distributions are chosen for demonstration. Artificial generation cannot establish external validity. Repeated patient cycles, eligibility/transfer selection, measurement and follow-up assumptions, residual confounding and multiple comparisons may distort association estimates. Some p-values are small because sample sizes are large; effect sizes must be considered. No individual prediction or medical recommendation is intended.'),
('14. Conclusion','The project demonstrates reproducible generation, data auditing, cleaning, linked-table validation, descriptive/statistical analysis, visualization and responsible communication. Observed synthetic gradients can motivate questions for monitoring, not clinical conclusions.'),
('15. Tools, Reproducibility and Disclaimer','Python 3; NumPy; pandas; SciPy; Matplotlib; Seaborn; scikit-learn; Jupyter; python-docx. From the project root install requirements and run `python src/generate_synthetic_data.py`, `python src/clean_and_preprocess.py`, `python src/statistical_analysis_and_charts.py`, then execute the notebook. The fixed seed is 42. The dataset contains no real patient records, and findings cannot guide clinical patient decisions.')]:
 doc.add_heading(title,1); para(body)
# References no fabricated external citations
doc.add_heading('References and Dataset Disclaimer',1); para('No external clinical references are cited; this report summarizes a generated educational dataset and locally executed code. Tool documentation may be consulted for software usage. Synthetic records and findings do not represent clinical evidence.')
doc.save(R/'IVF_Treatment_Outcome_Analysis_Report.docx')
# Internship description
text='''This project analyzes a fully synthetic dataset created for a fictional fertility clinic that wants to understand IVF treatment-cycle patterns, embryo grading, outcomes, and costs. The business challenge was to create an end-to-end analytics workflow that makes data quality visible, uses appropriate outcome denominators, and turns observed patterns into cautious operational questions without implying clinical evidence.\n\nUsing Python, NumPy, pandas, SciPy, Matplotlib, Seaborn, Jupyter, and python-docx, I generated six reproducible CSV tables with a fixed random seed. The linked dataset includes patient characteristics, more than five thousand cycle records, embryo morphology, follow-up outcomes, itemized costs, and a data dictionary. Controlled missingness, duplicates, inconsistent labels, orphan keys, and logical anomalies were introduced to demonstrate real data-cleaning practices.\n\nI profiled shapes, types, summaries, missing values, duplicates, categories, primary and foreign keys, and biological consistency. Cleaning standardized categories and dates, removed documented duplicates and invalid records, checked referential integrity, handled missing values, and reconciled cost totals to their components. An IQR-based audit identified distributional outliers without automatically deleting legitimate extreme values. Engineered fields supported age grouping, embryo-quality summaries, outcome indicators, and cost analyses.\n\nExploratory analysis described demographics, treatment modalities, oocyte development, embryo grading, outcome progression, and cost distributions. I distinguished rates per initiated cycle from rates per embryo transfer and reported implantation, clinical pregnancy, and live birth as separate endpoints. Statistical work included Pearson and Spearman correlations, chi-square tests with Cramer’s V, nonparametric group comparisons, and an exploratory multivariable logistic model. Eighteen high-resolution charts communicate these patterns.\n\nThe synthetic results show descriptive gradients by age and embryo-quality group, while selected measures such as BMI had weak, statistically insignificant associations. Costs varied by modality; these unadjusted comparisons are not causal. Recommendations focus on denominator-aware dashboards, follow-up completeness, cost-component monitoring, and better-designed future evaluation. This project strengthened my skills in reproducible data generation, relational validation, statistical interpretation, visualization, documentation, and ethical communication. No real patient records were used, and these results must not guide clinical decisions.'''
assert len(text.split())>=250
sd=Document(); sd.add_heading('Internship Project Submission Description',0); sd.add_paragraph('IVF Treatment Outcome Analysis & Success Rate Optimization Using Python'); sd.add_paragraph(text); sd.add_paragraph('Synthetic data disclaimer: no actual patient records; not for clinical decision-making.')
sd.save(R/'Internship_Submission_Description.docx')
print('Built notebook and DOCX deliverables; description words:',len(text.split()))
