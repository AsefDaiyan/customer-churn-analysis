# Customer Churn Prediction — IBM Telco Dataset

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange?logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)
![pandas](https://img.shields.io/badge/pandas-1.5+-150458?logo=pandas)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

A complete, end-to-end machine-learning project that predicts **which telecom customers are likely to cancel their subscription (churn)** — enabling the business to take proactive retention action before revenue is lost.

---

## Business Problem

Customer churn is one of the most costly challenges in the telecom sector. Acquiring a new subscriber costs 5–25× more than retaining an existing one. This project answers two key questions:

1. **What are the most important factors that drive churn?**
2. **Can we accurately predict which customers will churn before they do?**

---

## Dataset

| Property | Value |
|---|---|
| Source | [IBM Sample Data — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| Rows | 7,043 customers |
| Features | 21 (demographics, account info, services subscribed) |
| Target | `Churn` — Yes / No |

Key features include: `tenure`, `Contract`, `InternetService`, `MonthlyCharges`, `TotalCharges`, `PaymentMethod`, and 10+ service add-ons.

---

## Key Findings

1. **Month-to-month contract customers churn at 3–4× the rate** of annual contract holders — the single strongest churn predictor.
2. **Fibre Optic subscribers churn more** than DSL customers, suggesting price-value dissatisfaction at the premium tier.
3. **New customers (0–6 months tenure) are at the highest risk** — a structured onboarding programme would significantly reduce early churn.
4. **Higher monthly charges correlate with churn** — loyalty discounts for bills above ~USD 70/month are warranted.
5. The **Random Forest model achieves strong recall on churners**, making it practical as a weekly early-warning signal inside a CRM system.

---

## Project Structure

```
customer-churn-analysis/
├── data/
│   └── Telco-Customer-Churn.csv        ← IBM Telco dataset (7,043 rows)
├── notebooks/
│   └── customer_churn_analysis.ipynb   ← Full analysis notebook
├── src/
│   ├── __init__.py
│   ├── data_loader.py                  ← Load, clean, encode data
│   └── model_utils.py                  ← Evaluate, plot confusion matrix & feature importance
├── outputs/
│   └── figures/                        ← All generated PNG charts
├── README.md
├── requirements.txt
└── .gitignore
```

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/AhmedAsefDaiyanChowdhury/customer-churn-analysis.git
cd customer-churn-analysis
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter and run the notebook

```bash
jupyter notebook notebooks/customer_churn_analysis.ipynb
```

Run all cells top-to-bottom (`Kernel → Restart & Run All`). Generated figures will appear in `outputs/figures/`.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| pandas / numpy | Data manipulation |
| matplotlib / seaborn | Visualisation |
| scikit-learn | ML models & evaluation |
| Jupyter Notebook | Interactive analysis environment |

---

## Results Summary

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 0.7388 | 0.5052 | **0.7834** | 0.6143 |
| Random Forest | **0.7722** | **0.5582** | 0.6791 | **0.6128** |

> Models trained with `class_weight="balanced"` to handle the ~26% churn minority class.
> Logistic Regression achieves higher recall (fewer missed churners); Random Forest achieves higher overall accuracy.
> All metrics reproducible by running the notebook end-to-end (`random_state=42`).

---

## Author

**Ahmed Asef Daiyan Chowdhury**
MSc Computer Science & Technology (ML/AI)
Ulster University

Available for freelance data analysis, machine learning, and business intelligence projects on [Upwork](https://www.upwork.com) and [Fiverr](https://www.fiverr.com).

---

## License

This project is licensed under the MIT License.
