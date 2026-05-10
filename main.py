import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

from sklearn.decomposition import PCA


df = pd.read_csv("loan_dataset.csv")

df.columns = df.columns.str.strip()

print(df.shape)


drop_cols = ["Loan_ID"]

for col in drop_cols:
    if col in df.columns:
        df.drop(columns=col, inplace=True)


df["Total_Income"] = (
    df["ApplicantIncome"].fillna(0)
    + df["CoapplicantIncome"].fillna(0)
)

df["Estimated_EMI"] = (
    df["LoanAmount"].fillna(df["LoanAmount"].median())
    / df["Loan_Amount_Term"].fillna(360)
)


df["DTI_Ratio"] = (
    df["Estimated_EMI"]
    / (df["Total_Income"] + 1)
)


df["Loan_Income_Ratio"] = (
    df["LoanAmount"].fillna(0)
    / (df["Total_Income"] + 1)
)


df["Low_Credit_History"] = np.where(
    df["Credit_History"] == 0,
    1,
    0
)

numeric_features = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Total_Income",
    "Estimated_EMI",
    "DTI_Ratio",
    "Loan_Income_Ratio"
]

categorical_features = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area"
]


for col in numeric_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

for col in categorical_features:
    df[col] = df[col].astype("category")


numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])


X = df[numeric_features + categorical_features]

X_processed = preprocessor.fit_transform(X)


inertia_scores = []
silhouette_scores = []

cluster_range = range(2, 11)

for k in cluster_range:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(X_processed)

    inertia_scores.append(model.inertia_)

    score = silhouette_score(
        X_processed,
        model.labels_
    )

    silhouette_scores.append(score)


plt.figure(figsize=(10, 5))

plt.plot(cluster_range, inertia_scores, marker='o')

plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")

plt.grid(True)

plt.show()


plt.figure(figsize=(10, 5))

plt.plot(cluster_range, silhouette_scores, marker='o')

plt.title("Silhouette Scores")
plt.xlabel("Number of Clusters")
plt.ylabel("Score")

plt.grid(True)

plt.show()


optimal_k = 4

kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_processed)

df["Customer_Risk_Segment"] = clusters


pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_processed)

pca_df = pd.DataFrame({
    "PCA_1": X_pca[:, 0],
    "PCA_2": X_pca[:, 1],
    "Cluster": clusters
})

plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=pca_df,
    x="PCA_1",
    y="PCA_2",
    hue="Cluster",
    palette="Set1"
)

plt.title("Customer Risk Segmentation")

plt.show()


segment_summary = df.groupby(
    "Customer_Risk_Segment"
)[[
    "ApplicantIncome",
    "LoanAmount",
    "Total_Income",
    "DTI_Ratio",
    "Loan_Income_Ratio",
    "Credit_History"
]].mean()

print(segment_summary)


risk_mapping = {
    0: "Low Risk Customers",
    1: "Medium Risk Customers",
    2: "High Risk Customers",
    3: "Very High Risk Customers"
}

df["Risk_Label"] = df[
    "Customer_Risk_Segment"
].map(risk_mapping)


print(df["Risk_Label"].value_counts())
