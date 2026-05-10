import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import (classification_report,confusion_matrix,roc_auc_score,roc_curve)

df = pd.read_csv("PS_20174392719_1491204439457_log.csv")

fraud_type = df.groupby('type')['isFraud'].mean() * 100

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

df['type'].value_counts().plot(kind='bar', ax=axes[0], color='steelblue')
axes[0].set_title("Transaction Type Distribution")
fraud_by_type.plot(kind='bar', ax=axes[1], color='crimson')
axes[1].set_title("Fraud Rate by Transaction Type (%)")
plt.tight_layout()
plt.show()

df = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()

df['balance_drain']= ((df['newbalanceOrig'] == 0) & (df['oldbalanceOrg'] > 0)).astype(int)
df['amount_to_balance_ratio']= df['amount'] / (df['oldbalanceOrg'] + 1)
df['dest_balance_change'] = df['newbalanceDest'] - df['oldbalanceDest']
df['zero_origin_balance']= (df['oldbalanceOrg'] == 0).astype(int)
df['large_transaction'] = (df['amount'] > 200000).astype(int)

df = df.merge(df.groupby('nameOrig')['step'].count().reset_index().rename(columns={'step': 'transaction_velocity'}), on='nameOrig', how='left')
df = df.merge(df.groupby('nameOrig')['nameDest'].nunique().reset_index().rename(columns={'nameDest': 'unique_counterparties'}), on='nameOrig', how='left')
df = df.merge(df.groupby('nameOrig')['amount'].mean().reset_index().rename(columns={'amount': 'avg_transaction_amount'}), on='nameOrig', how='left')

def assign_lifecycle(step):
    if step <= 30:   return 'Onboarding'
    elif step <= 90: return 'Early_Stage'
    else:            return 'Portfolio'

df['lifecycle_stage'] = df['step'].apply(assign_lifecycle)

feature_cols = [
    'amount', 'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest',
    'balance_drain', 'amount_to_balance_ratio',
    'dest_balance_change', 'zero_origin_balance',
    'large_transaction', 'transaction_velocity',
    'unique_counterparties', 'avg_transaction_amount'
]

X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())
y = df['isFraud']

X_scaled = StandardScaler().fit_transform(X)

iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=42, n_jobs=-1)
iso.fit(X_scaled)

df['anomaly_score'] = iso.decision_function(X_scaled)
df['is_anomaly']    = (iso.predict(X_scaled) == -1).astype(int)

print(classification_report(y, df['is_anomaly']))

roc_auc = roc_auc_score(y, -df['anomaly_score'])
print(f"ROC-AUC: {roc_auc:.4f}")

fpr, tpr, _ = roc_curve(y, -df['anomaly_score'])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(fpr, tpr, color='crimson', label=f'AUC = {roc_auc:.4f}')
axes[0].plot([0,1],[0,1],'k--')
axes[0].set_title("ROC Curve")
axes[0].legend()

sns.heatmap(confusion_matrix(y, df['is_anomaly']), annot=True, fmt='d',
            cmap='Reds', ax=axes[1],
            xticklabels=['Normal','Fraud'],
            yticklabels=['Normal','Fraud'])
axes[1].set_title("Confusion Matrix")

sns.histplot(df[df['isFraud']==0]['anomaly_score'], bins=50, color='steelblue', label='Normal', alpha=0.6, ax=axes[2])
sns.histplot(df[df['isFraud']==1]['anomaly_score'], bins=50, color='crimson', label='Fraud', alpha=0.6, ax=axes[2])
axes[2].set_title("Anomaly Score Distribution")
axes[2].legend()
plt.tight_layout()
plt.show()

lifecycle_risk = df.groupby('lifecycle_stage').agg(
    total_transactions = ('isFraud', 'count'),
    fraud_cases        = ('isFraud', 'sum'),
    anomalies_detected = ('is_anomaly', 'sum'),
    avg_amount         = ('amount', 'mean')
).assign(fraud_rate=lambda x: (x['fraud_cases'] / x['total_transactions'] * 100).round(2))

print(lifecycle_risk)

X_pca = PCA(n_components=2).fit_transform(X_scaled)

pca_df = pd.DataFrame({'PCA_1': X_pca[:,0], 'PCA_2': X_pca[:,1],
                        'Label': df['is_anomaly'].map({0:'Normal', 1:'Suspected Mule'})})

plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_df, x='PCA_1', y='PCA_2', hue='Label',
                palette={'Normal':'steelblue','Suspected Mule':'crimson'}, alpha=0.5)
plt.title("PCA — Normal vs Suspected Mule Accounts")
plt.tight_layout()
plt.show()

sample_df = df[df['is_anomaly'] == 1].sample(min(500, df['is_anomaly'].sum()), random_state=42)

G = nx.DiGraph()
for _, row in sample_df.iterrows():
    G.add_edge(row['nameOrig'], row['nameDest'], weight=row['amount'])

betweenness = nx.betweenness_centrality(G, k=100)

centrality_df = pd.DataFrame({
    'account'              : list(betweenness.keys()),
    'betweenness_centrality': list(betweenness.values())
}).sort_values('betweenness_centrality', ascending=False)

print("\nTop 10 Suspected Mule Hubs:")
print(centrality_df.head(10))

communities = nx.community.greedy_modularity_communities(G.to_undirected())
print(f"\nMule Communities Detected : {len(communities)}")
print(f"Largest Network Size      : {max(len(c) for c in communities)} accounts")

plt.figure(figsize=(14, 10))
pos = nx.spring_layout(G, seed=42, k=0.5)
node_colors = ['crimson' if betweenness.get(n, 0) > 0.05 else 'steelblue' for n in G.nodes()]
nx.draw_networkx(G, pos, node_color=node_colors, node_size=30,
                 edge_color='gray', alpha=0.6, with_labels=False, arrows=True, arrowsize=5)
plt.title("Money Mule Transaction Network | Red = Mule Hub  Blue = Normal")
plt.axis('off')
plt.tight_layout()
plt.show()

print(f"""
Transactions Analysed : {len(df):,}
Anomalies Detected: {df['is_anomaly'].sum():,}
Confirmed Fraud Cases : {df['isFraud'].sum():,}
ROC-AUC Score : {roc_auc:.4f}
Graph Nodes (Accounts): {G.number_of_nodes():,}
Graph Edges (Txns): {G.number_of_edges():,}
Mule Communities : {len(communities)}
Largest Mule Network: {max(len(c) for c in communities)} accounts
""")
