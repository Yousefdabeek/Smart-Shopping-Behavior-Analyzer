import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from sklearn.ensemble import IsolationForest
#----------------------------------------------------------------
REQUIRED_COLUMNS = [
    "CustomerID",
    "InvoiceNo",
    "Description",
    "Quantity",
    "UnitPrice",
    "InvoiceDate"
]


def load_data(file_source):
    df = pd.read_csv(file_source)
    return df
#----------------------------------------------------------------
def preprocess_data(df):
    data = df.copy()

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data = data.dropna(subset=[
        "CustomerID",
        "InvoiceNo",
        "Description",
        "Quantity",
        "UnitPrice",
        "InvoiceDate"
    ])

    data["CustomerID"] = data["CustomerID"].astype(str)
    data["InvoiceNo"] = data["InvoiceNo"].astype(str)
    data["Description"] = data["Description"].astype(str)

    data["Quantity"] = pd.to_numeric(data["Quantity"], errors="coerce")
    data["UnitPrice"] = pd.to_numeric(data["UnitPrice"], errors="coerce")
    data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"], errors="coerce")

    data = data.dropna(subset=["Quantity", "UnitPrice", "InvoiceDate"])

    data = data[data["Quantity"] > 0]
    data = data[data["UnitPrice"] > 0]

    data["TotalPrice"] = data["Quantity"] * data["UnitPrice"]

    return data
#----------------------------------------------------------------
def create_customer_features(data):

    customer_df = data.groupby("CustomerID").agg({

        "TotalPrice": "sum",
        "Quantity": "sum",
        "InvoiceNo": "nunique",
        "Description": "nunique",
        "InvoiceDate": "max"

    }).reset_index()

    customer_df.columns = [
        "CustomerID",
        "TotalSpent",
        "TotalQuantity",
        "NumTransactions",
        "UniqueProducts",
        "LastPurchaseDate"
    ]

    reference_date = data["InvoiceDate"].max()

    customer_df["RecencyDays"] = (
        reference_date - customer_df["LastPurchaseDate"]
    ).dt.days

    customer_df["AvgTransactionValue"] = (
        customer_df["TotalSpent"] /
        customer_df["NumTransactions"]
    )

    customer_df = customer_df.drop(columns=["LastPurchaseDate"])

    return customer_df
#----------------------------------------------------------------
def scale_features(customer_df):

    feature_columns = [
        "TotalSpent",
        "TotalQuantity",
        "NumTransactions",
        "UniqueProducts",
        "RecencyDays",
        "AvgTransactionValue"
    ]

    X = customer_df[feature_columns]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled, feature_columns
#----------------------------------------------------------------
def calculate_sse(X_scaled, max_k=10):

    sse = []

    for k in range(1, max_k + 1):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(X_scaled)

        sse.append(model.inertia_)

    return sse


def choose_best_k(X_scaled, max_k=10):

    best_k = 2
    best_score = -1

    for k in range(2, max_k + 1):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(X_scaled)

        score = silhouette_score(X_scaled, labels)

        if score > best_score:
            best_score = score
            best_k = k

    return best_k, best_score

#----------------------------------------------------------------
def run_kmeans(customer_df, X_scaled, fixed_k=None):

    if fixed_k is not None and fixed_k >= 2:
        best_k = fixed_k
        silhouette = None
    else:
        max_k = min(10, len(customer_df) - 1)
        best_k, silhouette = choose_best_k(X_scaled, max_k=max_k)

    model = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=10
    )

    cluster_labels = model.fit_predict(X_scaled)

    clustered_customers = customer_df.copy()
    clustered_customers["cluster"] = cluster_labels

    cluster_summary = clustered_customers.groupby("cluster").agg({
        "CustomerID": "count",
        "TotalSpent": "mean",
        "TotalQuantity": "mean",
        "NumTransactions": "mean",
        "UniqueProducts": "mean",
        "RecencyDays": "mean",
        "AvgTransactionValue": "mean"
    }).reset_index()

    cluster_summary = cluster_summary.rename(columns={
        "CustomerID": "NumberOfCustomers"
    })

    return clustered_customers, cluster_summary, best_k, silhouette
#----------------------------------------------------------------
def run_pca(clustered_customers, X_scaled):

    pca = PCA(n_components=2)

    pca_result = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(
        pca_result,
        columns=["PC1", "PC2"]
    )

    pca_df["CustomerID"] = clustered_customers["CustomerID"].values
    pca_df["cluster"] = clustered_customers["cluster"].values

    explained_variance = pca.explained_variance_ratio_

    return pca_df, explained_variance
#----------------------------------------------------------------
def run_apriori(data, min_support=0.02, min_confidence=0.3):

    basket = data.groupby(["InvoiceNo", "Description"])["Quantity"].sum().unstack().fillna(0)

    basket = (basket > 0).astype(int)

    frequent_items = apriori(
        basket,
        min_support=min_support,
        use_colnames=True
    )

    if frequent_items.empty:
        return pd.DataFrame()

    rules = association_rules(
        frequent_items,
        metric="confidence",
        min_threshold=min_confidence
    )

    if rules.empty:
        return pd.DataFrame()

    rules = rules.sort_values(
        by=["lift", "confidence"],
        ascending=False
    )

    rules = rules[[
        "antecedents",
        "consequents",
        "support",
        "confidence",
        "lift"
    ]]

    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(list(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(list(x)))

    return rules
#----------------------------------------------------------------
def run_anomaly_detection(clustered_customers):

    feature_columns = [
        "TotalSpent",
        "TotalQuantity",
        "NumTransactions",
        "UniqueProducts",
        "RecencyDays",
        "AvgTransactionValue"
    ]

    X = clustered_customers[feature_columns]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    labels = model.fit_predict(X_scaled)

    anomaly_results = clustered_customers.copy()

    anomaly_results["Anomaly"] = labels

    anomaly_results["AnomalyLabel"] = anomaly_results["Anomaly"].map({
        1: "Normal",
        -1: "Anomaly"
    })

    anomaly_results["AnomalyScore"] = model.decision_function(X_scaled)

    anomalies = anomaly_results[
        anomaly_results["AnomalyLabel"] == "Anomaly"
    ]

    return anomaly_results, anomalies
#----------------------------------------------------------------
def generate_insights(
    clustered_customers,
    rules,
    anomalies,
    explained_variance
):

    insights = []

    highest_spending_cluster = (
        clustered_customers.groupby("cluster")["TotalSpent"]
        .mean()
        .idxmax()
    )

    insights.append(
        f"cluster {highest_spending_cluster} contains the highest-spending customers."
    )

    if not rules.empty:

        top_rule = rules.iloc[0]

        insights.append(
            f"The strongest product rule is "
            f"{top_rule['antecedents']} -> "
            f"{top_rule['consequents']}."
        )

    else:

        insights.append(
            "No strong association rules were found."
        )

    retained_variance = (
        explained_variance[0] +
        explained_variance[1]
    ) * 100

    insights.append(
        f"PCA retained {retained_variance:.1f}% of the original variance."
    )

    insights.append(
        f"{len(anomalies)} anomalous customers were detected."
    )

    return insights
#----------------------------------------------------------------
def run_full_analysis(
    file_source,
    fixed_k=None,
    min_support=0.02,
    min_confidence=0.3
):

    raw_data = load_data(file_source)

    clean_data = preprocess_data(raw_data)

    customer_features = create_customer_features(clean_data)

    X_scaled, feature_columns = scale_features(customer_features)

    sse_values = calculate_sse(
        X_scaled,
        max_k=min(10, len(customer_features))
    )

    clustered_customers, cluster_summary, best_k, silhouette = run_kmeans(
        customer_features,
        X_scaled,
        fixed_k=fixed_k
    )

    pca_results, explained_variance = run_pca(
        clustered_customers,
        X_scaled
    )

    association_rules_results = run_apriori(
        clean_data,
        min_support=min_support,
        min_confidence=min_confidence
    )

    anomaly_results, anomalies = run_anomaly_detection(
        clustered_customers
    )

    insights = generate_insights(
        clustered_customers,
        association_rules_results,
        anomalies,
        explained_variance
    )

    return {
        "raw_data": raw_data,
        "clean_data": clean_data,
        "customer_features": customer_features,
        "feature_columns": feature_columns,
        "sse_values": sse_values,
        "customers": clustered_customers,
        "cluster_summary": cluster_summary,
        "best_k": best_k,
        "silhouette_score": silhouette,
        "pca": pca_results,
        "explained_variance": explained_variance,
        "rules": association_rules_results,
        "anomaly_results": anomaly_results,
        "anomalies": anomalies,
        "insights": insights
    }
#----------------------------------------------------------------
