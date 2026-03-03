# CreditIQ — Explainable Credit Risk Scoring System

Explainable credit scoring system — XGBoost + SHAP + PostgreSQL + Streamlit. AUC 0.7741 on 300K loan applications.

---

## Overview

CreditIQ is an end-to-end credit risk intelligence platform built on the Home Credit Default Risk dataset. It predicts the probability that a loan applicant will default, explains every prediction using SHAP values, assigns a 300–850 credit score, and surfaces decisions through a live web application and Power BI dashboard.

The system covers the full data science lifecycle: data ingestion into a normalized PostgreSQL database, feature engineering across six relational tables, XGBoost model training with Optuna hyperparameter tuning, SHAP explainability analysis, and deployment as an interactive Streamlit app.

---

## Results

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.7741 |
| Recall (defaulters) | 56.2% |
| Precision (defaulters) | 21.5% |
| Classification threshold | 0.59 |
| Portfolio default rate | 8.07% |
| HIGH RISK clients identified | 12,369 |
| Total clients scored | 61,502 |

---

## Architecture

```
PostgreSQL 18 (credit_risk_db)
    6 tables · 15M+ rows
    client | loan | bureau | previous_application
    installments_payments | risk_score
        |
        | SQLAlchemy
        v
ML Pipeline
    XGBoost + Optuna · 55 features
    scale_pos_weight = 11.42
        |
        | SHAP TreeExplainer
        v
XAI Layer
    SHAP values stored in PostgreSQL
    Credit score 300-850 · Risk tier · Waterfall explanations
        |
        |-----> Power BI Dashboard (2 pages, live connection)
        |-----> Streamlit App (3 pages, live connection)
```

---

## Project Structure

```
.
├── notebooks/
│   ├── 01-EDA.ipynb
│   ├── 02-Preprocessing.ipynb
│   ├── 03-Modeling.ipynb
│   └── 04-XAI_Explainability.ipynb
├── sql/
│   ├── week2_sql.sql          # Schema + data loading
│   └── week3_sql.sql          # Advanced queries + views + KPIs
├── models/
│   ├── xgboost_model.joblib
│   ├── best_threshold.joblib
│   ├── best_params.joblib
│   ├── categorical_cols.joblib
│   └── numeric_cols.joblib
├── reports/
│   ├── shap_importance.png
│   ├── shap_summary.png
│   ├── shap_dependence.png
│   ├── shap_waterfall_high.png
│   ├── shap_waterfall_low.png
│   └── shap_by_tier.png
├── app.py
├── requirements.txt
└── README.md
```

---

## Setup

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/CreditIQ-explainable-credit-risk-scoring-system.git
cd CreditIQ-explainable-credit-risk-scoring-system
```

**2. Create virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up PostgreSQL**

- Install PostgreSQL 18 and create a database named `credit_risk_db`
- Download the [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk/data) dataset from Kaggle
- Run `sql/week2_sql.sql` to create the schema and load data
- Run `sql/week3_sql.sql` to create analytical views and KPIs

**5. Run the notebooks in order**
```
01-EDA.ipynb
02-Preprocessing.ipynb
03-Modeling.ipynb
04-XAI_Explainability.ipynb
```

**6. Launch the app**
```bash
streamlit run app.py
```

---

## App Pages

**Scoring Terminal**
Input a client profile and get an instant credit decision — score, risk tier, approve/review/reject verdict, and a SHAP waterfall explanation.

**Portfolio Intelligence**
Live KPI dashboard pulling from PostgreSQL views. Default rates by age bracket, income type risk profiles, score band distributions.

**Client Lookup**
Search any of the 61,502 scored clients by ID. Returns full risk profile, credit score, SHAP contributors, and decision recommendation.

---

## Key SHAP Findings

EXT_SOURCE_2 (external bureau score) is the dominant feature with a mean absolute SHAP value of 0.142. Clients below the portfolio median on this feature default at 34% versus 4% for those above 0.6 — an 8.5x difference driven by a single variable.

AGE_YEARS shows a monotonic protective gradient above age 30. The 18-24 bracket defaults at 10.2%, which is 2.3x the portfolio average.

Bureau enrichment from PostgreSQL contributes measurably to model performance. BUR_TOTAL_DEBT ranks 4th globally with a mean absolute SHAP value of 0.063, validating the multi-table data architecture.

---

## SQL Views

| View | Purpose |
|------|---------|
| v_client_risk_profile | Full client profile with ML scores and SHAP values |
| v_kpi_summary | Single-row portfolio aggregation across 9 KPIs |
| v_risk_by_segment | Age bracket x income type x risk tier breakdown |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Database | PostgreSQL 18 |
| ML model | XGBoost 3.2 + Optuna 4.7 |
| Explainability | SHAP 0.50 |
| Data processing | pandas 3.0, scikit-learn 1.8 |
| Web app | Streamlit |
| BI dashboard | Power BI Desktop |
| ORM | SQLAlchemy 2.0 + psycopg2 |

---

## Data

Home Credit Default Risk — Kaggle competition dataset.
307,511 loan applications with bureau, previous application, and installment payment history.

Data files are not included in this repository. Download from Kaggle and follow the setup instructions above.

---

## Author

Youssef Dihaji
Data Science — INPT (Institut National des Postes et Télécommunications)
March 2026