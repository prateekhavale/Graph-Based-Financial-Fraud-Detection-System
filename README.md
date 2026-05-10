# Money Mule Detection & Financial Fraud Analytics

Graph-based financial fraud detection system for identifying suspicious transactions, money mule accounts, and fraud communities using anomaly detection and network analytics.

## Overview

This project combines:
- Unsupervised fraud detection
- Behavioral feature engineering
- Transaction network analysis
- Community detection
- Financial risk analytics

The system uses Isolation Forest for anomaly detection and NetworkX for graph-based fraud investigation. :contentReference[oaicite:0]{index=0}

---

## Key Features

- Fraud transaction analysis
- Behavioral risk feature engineering
- Isolation Forest anomaly detection
- PCA-based anomaly visualization
- Money mule hub detection
- Fraud community detection
- Transaction network analysis
- Lifecycle-based risk segmentation

---

## Tech Stack

### Programming & Analytics
- Python
- Pandas
- NumPy

### Machine Learning
- Scikit-learn
- Isolation Forest
- PCA

### Visualization
- Matplotlib
- Seaborn

### Graph Analytics
- NetworkX

---

## Features Engineered

- Balance Drain Detection
- Transaction Velocity
- Amount-to-Balance Ratio
- Large Transaction Detection
- Destination Balance Change
- Unique Counterparty Analysis
- Average Transaction Amount

---

## Workflow

```text
Transaction Data
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
Anomaly Detection
      ↓
Fraud Scoring
      ↓
PCA Visualization
      ↓
Graph Construction
      ↓
Mule Hub Detection
      ↓
Community Detection
```

---

## Model Evaluation

Metrics used:
- ROC-AUC Score
- Confusion Matrix
- Classification Report

---

## Graph Analytics

Implemented:
- Betweenness Centrality
- Community Detection
- Transaction Network Visualization

Used for:
- detecting mule hubs
- identifying suspicious transaction communities
- analyzing fraud networks

---

## Dataset

Dataset:
```text
PS_20174392719_1491204439457_log.csv
```

Contains:
- transaction details
- sender/receiver accounts
- balances
- fraud labels
- transaction types

---

## How to Run

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn networkx
```

### Run Project

```bash
python fraud_detection.py
```

---

## Future Improvements

- Graph Neural Networks (GNNs)
- Real-time fraud detection
- Explainable AI for fraud analysis
- Neo4j graph integration
- Deep anomaly detection
```
