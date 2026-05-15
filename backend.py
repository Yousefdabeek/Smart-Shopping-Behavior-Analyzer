import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


REQUIRED_COLUMNS = [
    "CustomerID",
    "InvoiceNo",
    "Description",
    "Quantity",
    "UnitPrice",
    "InvoiceDate"
]


def load_data(file_source):
    return pd.read_csv(file_source)


def preprocess_data(df):
    data = df.copy()

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data = data.dropna(subset=REQUIRED_COLUMNS)

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

    if data.empty:
        raise ValueError("No valid data remains after preprocessing.")

    return data


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
        customer_df["TotalSpent"] / customer_df["NumTransactions"]
    )

    customer_df = customer_df.drop(columns=["LastPurchaseDate"])

    return customer_df


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


def calculate_bic_scores(X_scaled, max_k=10):
    bic_scores = []

    for k in range(1, max_k + 1):
        model = GaussianMixture(
            n_components=k,
            random_state=42
        )

        model.fit(X_scaled)
        bic_scores.append(model.bic(X_scaled))

    return bic_scores


def choose_best_k(X_scaled, max_k=10):
    best_k = 2
    best_score = -1

    for k in range(2, max_k + 1):
        model = GaussianMixture(
            n_components=k,
            random_state=42
        )

        labels = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)

        if score > best_score:
            best_score = score
            best_k = k

    return best_k, best_score


def run_gmm(customer_df, X_scaled, fixed_k=None):
    if len(customer_df) < 3:
        grouped_customers = customer_df.copy()
        grouped_customers["CustomerGroup"] = 0
        grouped_customers["GroupConfidence"] = 1.0

        group_summary = grouped_customers.groupby("CustomerGroup").agg({
            "CustomerID": "count",
            "TotalSpent": "mean",
            "TotalQuantity": "mean",
            "NumTransactions": "mean",
            "UniqueProducts": "mean",
            "RecencyDays": "mean",
            "AvgTransactionValue": "mean",
            "GroupConfidence": "mean"
        }).reset_index()

        group_summary = group_summary.rename(columns={
            "CustomerID": "NumberOfCustomers"
        })

        return grouped_customers, group_summary, 1, None

    if fixed_k is not None and fixed_k >= 2:
        best_k = min(int(fixed_k), len(customer_df) - 1)
        group_quality = None
    else:
        max_k = min(10, len(customer_df) - 1)
        best_k, group_quality = choose_best_k(X_scaled, max_k=max_k)

    model = GaussianMixture(
        n_components=best_k,
        random_state=42
    )

    group_labels = model.fit_predict(X_scaled)
    group_confidence = model.predict_proba(X_scaled).max(axis=1)

    grouped_customers = customer_df.copy()
    grouped_customers["CustomerGroup"] = group_labels
    grouped_customers["GroupConfidence"] = group_confidence

    group_summary = grouped_customers.groupby("CustomerGroup").agg({
        "CustomerID": "count",
        "TotalSpent": "mean",
        "TotalQuantity": "mean",
        "NumTransactions": "mean",
        "UniqueProducts": "mean",
        "RecencyDays": "mean",
        "AvgTransactionValue": "mean",
        "GroupConfidence": "mean"
    }).reset_index()

    group_summary = group_summary.rename(columns={
        "CustomerID": "NumberOfCustomers"
    })

    return grouped_customers, group_summary, best_k, group_quality


def run_tsne(grouped_customers, X_scaled):
    n_samples = len(grouped_customers)

    if n_samples < 3:
        customer_map = pd.DataFrame({
            "MapX": [0] * n_samples,
            "MapY": [0] * n_samples,
            "CustomerID": grouped_customers["CustomerID"].values,
            "CustomerGroup": grouped_customers["CustomerGroup"].values
        })

        return customer_map

    perplexity_value = min(30, max(2, (n_samples - 1) // 3))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity_value,
        random_state=42,
        init="pca",
        learning_rate="auto"
    )

    tsne_result = tsne.fit_transform(X_scaled)

    customer_map = pd.DataFrame(
        tsne_result,
        columns=["MapX", "MapY"]
    )

    customer_map["CustomerID"] = grouped_customers["CustomerID"].values
    customer_map["CustomerGroup"] = grouped_customers["CustomerGroup"].values

    return customer_map


def run_apriori(data, min_support=0.02, min_confidence=0.3):
    basket = (
        data.groupby(["InvoiceNo", "Description"])["Quantity"]
        .sum()
        .unstack()
        .fillna(0)
    )

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
    ]].copy()

    rules["antecedents"] = rules["antecedents"].apply(
        lambda x: ", ".join(sorted(list(x)))
    )

    rules["consequents"] = rules["consequents"].apply(
        lambda x: ", ".join(sorted(list(x)))
    )

    rules = rules.rename(columns={
        "antecedents": "ProductsBought",
        "consequents": "AlsoBought",
        "support": "Frequency",
        "confidence": "Chance",
        "lift": "RelationshipStrength"
    })

    return rules


def run_anomaly_detection(grouped_customers):
    feature_columns = [
        "TotalSpent",
        "TotalQuantity",
        "NumTransactions",
        "UniqueProducts",
        "RecencyDays",
        "AvgTransactionValue"
    ]

    X = grouped_customers[feature_columns]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    labels = model.fit_predict(X_scaled)

    anomaly_results = grouped_customers.copy()

    anomaly_results["UnusualFlag"] = labels

    anomaly_results["CustomerStatus"] = anomaly_results["UnusualFlag"].map({
        1: "Normal",
        -1: "Unusual"
    })

    anomaly_results["UnusualScore"] = model.decision_function(X_scaled)

    unusual_customers = anomaly_results[
        anomaly_results["CustomerStatus"] == "Unusual"
    ]

    return anomaly_results, unusual_customers


def generate_insights(grouped_customers, rules, unusual_customers):
    insights = []

    highest_spending_group = (
        grouped_customers.groupby("CustomerGroup")["TotalSpent"]
        .mean()
        .idxmax()
    )

    insights.append(
        f"Customer Group {highest_spending_group} has the highest average spending."
    )

    if not rules.empty:
        top_rule = rules.iloc[0]

        insights.append(
            f"Customers who buy {top_rule['ProductsBought']} often also buy {top_rule['AlsoBought']}."
        )
    else:
        insights.append(
            "No strong product-pair pattern was found with the current settings."
        )

    insights.append(
        "The customer map shows which customers have similar shopping behavior."
    )

    insights.append(
        f"{len(unusual_customers)} unusual customers were detected."
    )

    return insights


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

    bic_scores = calculate_bic_scores(
        X_scaled,
        max_k=min(10, len(customer_features))
    )

    grouped_customers, group_summary, best_k, group_quality = run_gmm(
        customer_features,
        X_scaled,
        fixed_k=fixed_k
    )

    customer_map = run_tsne(
        grouped_customers,
        X_scaled
    )

    product_rules = run_apriori(
        clean_data,
        min_support=min_support,
        min_confidence=min_confidence
    )

    anomaly_results, unusual_customers = run_anomaly_detection(
        grouped_customers
    )

    insights = generate_insights(
        grouped_customers,
        product_rules,
        unusual_customers
    )

    return {
        "raw_data": raw_data,
        "clean_data": clean_data,
        "customer_features": customer_features,
        "feature_columns": feature_columns,
        "bic_scores": bic_scores,
        "customers": grouped_customers,
        "group_summary": group_summary,
        "best_k": best_k,
        "group_quality_score": group_quality,
        "customer_map": customer_map,
        "product_rules": product_rules,
        "anomaly_results": anomaly_results,
        "unusual_customers": unusual_customers,
        "business_insights": insights
    }