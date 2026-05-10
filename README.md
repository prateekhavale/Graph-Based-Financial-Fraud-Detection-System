# Customer-Risk-Segmentation

## Overview

This project builds an industry-style Customer Risk Segmentation system for the banking and finance domain.

The goal is to group customers into different risk categories using:
- customer income,
- loan details,
- debt ratios,
- credit history,
- financial behavior.

The project uses:
- data preprocessing,
- feature engineering,
- KMeans clustering,
- PCA visualization.

---

# Objective

The main objective is to:

- identify risky customers,
- group similar borrowers,
- improve credit decision making,
- support portfolio monitoring.

---

# Workflow

```text
Raw Customer Data
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Preprocessing
        ↓
Feature Scaling & Encoding
        ↓
KMeans Clustering
        ↓
Customer Risk Segmentation
        ↓
PCA Visualization
        ↓
Business Interpretation
```

---

# Dataset Features

| Feature | Description |
|---|---|
| ApplicantIncome | Applicant income |
| CoapplicantIncome | Co-applicant income |
| LoanAmount | Loan amount |
| Loan_Amount_Term | Loan duration |
| Credit_History | Credit repayment history |
| Gender | Gender |
| Married | Marital status |
| Education | Education level |
| Self_Employed | Employment status |
| Property_Area | Area category |

---

# Feature Engineering

Additional financial risk features created:

| Feature | Purpose |
|---|---|
| Total_Income | Combined income |
| Estimated_EMI | Estimated repayment burden |
| DTI_Ratio | Debt-to-Income ratio |
| Loan_Income_Ratio | Loan burden relative to income |
| Low_Credit_History | Weak credit history indicator |

---

# Technologies Used

## Language
- Python

## Libraries
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

# Machine Learning Techniques

## KMeans Clustering
Used for:
- customer grouping,
- risk segmentation.

## PCA (Principal Component Analysis)
Used for:
- cluster visualization,
- dimensionality reduction.

## Silhouette Score
Used for:
- evaluating cluster quality.

---

# Preprocessing Pipeline

## Numeric Features
- Median Imputation
- Standard Scaling

## Categorical Features
- Most Frequent Imputation
- One-Hot Encoding

---

# Risk Segments

The model creates customer groups such as:

| Segment | Meaning |
|---|---|
| Low Risk Customers | Financially stable |
| Medium Risk Customers | Moderate risk |
| High Risk Customers | Higher repayment risk |
| Very High Risk Customers | Weak financial profile |

---

# Industry Use Cases

- Credit Risk Assessment
- Loan Approval Analysis
- Customer Portfolio Monitoring
- Risk-Based Loan Pricing
- Collection Strategy
- Banking Analytics

---

# How to Run the Project

## 1. Clone Repository

```bash
git clone <repository-url>
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Add Dataset

Place:

```text
loan_dataset.csv
```

inside the project folder.

## 4. Run Project

```bash
python customer_risk_segmentation.py
```

---

# Expected Outputs

The project generates:
- customer risk clusters,
- PCA visualization plots,
- segment summary tables,
- risk labels.

---

# Example Business Interpretation

## Low Risk Customers
- strong income,
- low debt burden,
- good credit history.

## High Risk Customers
- weaker income,
- high leverage,
- poor credit history.

---

# Future Improvements

Possible enhancements:
- DBSCAN clustering
- Gaussian Mixture Models
- Autoencoder-based clustering
- Behavioral scoring
- Real-time risk monitoring
- Explainable AI (SHAP)

---

# Conclusion

This project demonstrates an industry-style implementation of Customer Risk Segmentation in Banking & Finance using:
- financial feature engineering,
- clustering techniques,
- preprocessing pipelines,
- and business risk interpretation.
