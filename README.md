# Strategic Targeting for Financial Inclusion: Optimized Outreach ROI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/sklearn-latest-orange.svg)](https://scikit-learn.org/stable/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview
This project addresses the challenge of financial exclusion by leveraging machine learning to optimize marketing outreach. The goal is to identify the **top 20% of unbanked individuals** most likely to open a bank account when targeted with a mobile banking awareness campaign.

By transitioning from broad, imprecise targeting to a data-driven approach, financial institutions can significantly improve conversion rates and maximize the Return on Investment (ROI) for outreach initiatives.

---

## 💼 Business Objective
Narrowing the "unbanked gap" requires efficient resource allocation.
- **Problem:** Traditional marketing to all unbanked individuals is expensive and yields low conversion.
- **Solution:** A predictive system that ranks individuals based on their propensity to adopt banking services.
- **Success Metric:** Improvement in marketing outreach efficiency (Conversion Rate vs. Control Group) and optimized cost-per-acquisition.

---

## 🏗️ Pipeline Architecture
The project implements a production-grade, end-to-end ML pipeline:

1.  **Raw Data Ingestion:** Automated download and integrity validation (SHA-256) of the Global Findex dataset.
2.  **Data Extraction:** Systematic extraction and organization of microdata.
3.  **Preprocessing & Feature Engineering:** Clean-first, signal-driven strategy focusing on actionable demographic and behavioral indicators.
4.  **Modeling:** Implementation of a Support Vector Machine (SVM) pipeline optimized for high-precision targeting.
5.  **Explainability:** Integration of **SHAP (SHapley Additive exPlanations)** to provide transparent, defensible reasons for individual predictions.

---

## 📊 Dataset: Global Findex Database 2025
The analysis uses microdata from the **Global Findex Database 2025** provided by the World Bank.
- **Source:** [World Bank Microdata Catalog](https://microdata.worldbank.org/catalog/7860/get-microdata)
- **Focus:** Individuals identified as currently unbanked.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- `pip` package manager

### Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ProfFausat/Financial-Inclusion-Project
    cd v2_production/notebooks
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Usage
- **Notebook Execution:** Open `Financial_Inclusion_Project_v2.ipynb` to view the full workflow, from data ingestion to model evaluation.
- **Model Inference:** The trained pipeline is serialized as `svm_unbanked_pipeline.pkl` and can be loaded using `joblib`.

---

## 📈 Model Performance & Evaluation
Rather than aggregate accuracy, this project prioritizes:
- **Precision at Top 20%:** Ensuring the highest quality leads for the marketing team.
- **Recall–Precision Trade-offs:** Demonstrating how model choice adapts to outreach economics and operating thresholds.
- **Fairness Auditing:** Ensuring the targeting system does not introduce or exacerbate socio-economic biases.

---

## 👤 Author
**Fausat Ibrahim**  
*Independent Data Scientist*

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
