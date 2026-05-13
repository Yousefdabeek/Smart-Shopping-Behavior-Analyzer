import pandas as pd
import plotly.express as px
import streamlit as st
import backend


st.set_page_config(
    page_title="Smart Shopping Behavior Analyzer",
    page_icon="🛒",
    layout="wide"
)


st.markdown("""
<style>
.stApp {
    background-color: #0f172a;
    color: white;
}

.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #1f2937;
}

.hero-card {
    background: linear-gradient(135deg, #111827, #1e293b);
    padding: 24px;
    border-radius: 18px;
    border: 1px solid #334155;
    margin-bottom: 20px;
    color: white;
}

.section-title {
    font-size: 24px;
    font-weight: 700;
    margin-top: 10px;
    margin-bottom: 15px;
    color: white;
}

div[data-testid="metric-container"] {
    background-color: #111827;
    border: 1px solid #334155;
    padding: 16px;
    border-radius: 14px;
}

div[data-testid="stMetricValue"] {
    color: white;
    font-size: 28px;
    font-weight: 700;
}

div[data-testid="stMetricLabel"] {
    color: #cbd5e1;
}

.stDataFrame {
    border-radius: 12px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)


def sidebar_controls():
    st.sidebar.title("Smart Shopping Analyzer")
    st.sidebar.write("Upload a CSV file and run the analysis.")

    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
    run_button = st.sidebar.button("Run Analysis", type="primary")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Optional Settings")

    cluster_k = st.sidebar.number_input(
        "Number of clusters (0 = automatic)",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )

    min_support = st.sidebar.slider(
        "Minimum support",
        min_value=0.005,
        max_value=0.20,
        value=0.02,
        step=0.005
    )

    min_confidence = st.sidebar.slider(
        "Minimum confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.30,
        step=0.05
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Instructions")
    st.sidebar.markdown("1. Upload CSV\n2. Click Run Analysis\n3. Explore results")

    return uploaded_file, run_button, cluster_k, min_support, min_confidence


def show_header():
    st.title("🛒 Smart Shopping Behavior Analyzer")

    st.caption(
        "An interactive unsupervised data mining dashboard for discovering customer segments, "
        "product relationships, and unusual shopping behavior."
    )

    st.markdown("""
    <div class="hero-card">
    <h3>Discover hidden customer behavior patterns using unsupervised learning</h3>
    <p>This dashboard analyzes shopping transactions using:</p>
    <ul>
        <li>Gaussian Mixture Model clustering</li>
        <li>t-SNE visualization</li>
        <li>Association Rule Mining</li>
        <li>Anomaly Detection</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)


def show_overview(results):
    st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)

    clean_data = results["clean_data"]
    raw_data = results["raw_data"]

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Rows", f"{len(clean_data):,}")
    col2.metric("Customers", f"{clean_data['CustomerID'].nunique():,}")
    col3.metric("Invoices", f"{clean_data['InvoiceNo'].nunique():,}")
    col4.metric("Products", f"{clean_data['Description'].nunique():,}")
    col5.metric("Revenue", f"${clean_data['TotalPrice'].sum():,.0f}")

    st.markdown('<div class="section-title">Data Preview</div>', unsafe_allow_html=True)

    st.dataframe(raw_data.head(12), use_container_width=True, height=320)

    left, right = st.columns([2, 1])

    with left:
        st.markdown("### Detected Columns")
        st.write(", ".join(raw_data.columns))

    with right:
        st.markdown("### Missing Values")
        missing_values = raw_data.isna().sum().reset_index()
        missing_values.columns = ["Column", "MissingCount"]
        st.dataframe(missing_values, use_container_width=True, height=220)


def show_clustering(results):
    st.markdown('<div class="section-title">GMM Clustering Results</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    col1.metric("Number of Clusters", results["best_k"])

    if results["silhouette_score"] is not None:
        col2.metric("Silhouette Score", f"{results['silhouette_score']:.4f}")
    else:
        col2.metric("Silhouette Score", "Not available")

    clustered_customers = results["customers"]

    cluster_counts = (
        clustered_customers["cluster"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    cluster_counts.columns = ["cluster", "Customers"]

    fig = px.bar(
        cluster_counts,
        x="cluster",
        y="Customers",
        color="cluster",
        text="Customers",
        title="GMM Cluster Distribution"
    )

    fig.update_layout(
        showlegend=False,
        plot_bgcolor="#111827",
        paper_bgcolor="#111827",
        font_color="white"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Cluster Summary")

    st.dataframe(results["cluster_summary"], use_container_width=True, height=280)

    st.download_button(
        "Download Clustered Customers CSV",
        data=results["customers"].to_csv(index=False),
        file_name="clustered_customers.csv",
        mime="text/csv"
    )


def show_tsne(results):
    st.markdown('<div class="section-title">t-SNE Visualization</div>', unsafe_allow_html=True)

    tsne_df = results["tsne"]

    fig = px.scatter(
        tsne_df,
        x="TSNE1",
        y="TSNE2",
        color=tsne_df["cluster"].astype(str),
        hover_data=["CustomerID"],
        title="t-SNE Customer Visualization",
        labels={"color": "cluster"}
    )

    fig.update_layout(
        plot_bgcolor="#111827",
        paper_bgcolor="#111827",
        font_color="white"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info("t-SNE reduces customer behavior features into 2D for visualization.")

    st.download_button(
        "Download t-SNE Results CSV",
        data=results["tsne"].to_csv(index=False),
        file_name="tsne_results.csv",
        mime="text/csv"
    )


def show_association_rules(results):
    st.markdown('<div class="section-title">Association Rules</div>', unsafe_allow_html=True)

    rules = results["rules"]

    st.write("These rules show products that are frequently purchased together.")

    if rules.empty:
        st.warning("No association rules found. Try lowering minimum support or confidence.")
        return

    top_rule = rules.iloc[0]

    st.markdown(
        f"""
        <div class="hero-card">
        <b>Strongest Rule</b><br><br>
        Customers who buy <b>{top_rule['antecedents']}</b>
        are likely to also buy <b>{top_rule['consequents']}</b>.<br><br>
        Support: {top_rule['support']:.3f} |
        Confidence: {top_rule['confidence']:.3f} |
        Lift: {top_rule['lift']:.3f}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.dataframe(rules.head(20), use_container_width=True)

    st.download_button(
        "Download Association Rules CSV",
        data=rules.to_csv(index=False),
        file_name="association_rules.csv",
        mime="text/csv"
    )


def show_anomalies(results):
    st.markdown('<div class="section-title">Anomaly Detection</div>', unsafe_allow_html=True)

    anomalies = results["anomalies"]

    st.metric("Anomalous Customers", len(anomalies))

    st.write(
        "Anomalies represent customers with unusual shopping behavior compared to the rest of the dataset."
    )

    if anomalies.empty:
        st.info("No anomalies were detected.")
        return

    display_columns = [
        "CustomerID",
        "TotalSpent",
        "NumTransactions",
        "AvgTransactionValue",
        "RecencyDays",
        "AnomalyScore"
    ]

    st.dataframe(
        anomalies[display_columns].sort_values("AnomalyScore"),
        use_container_width=True,
        height=320
    )

    st.download_button(
        "Download Anomalies CSV",
        data=anomalies[display_columns].to_csv(index=False),
        file_name="anomalies.csv",
        mime="text/csv"
    )


def show_insights(results):
    st.markdown('<div class="section-title">Insights</div>', unsafe_allow_html=True)

    for insight in results["insights"]:
        st.markdown(f"- {insight}")


def main():
    show_header()

    uploaded_file, run_button, cluster_k, min_support, min_confidence = sidebar_controls()

    if run_button:
        if uploaded_file is None:
            st.error("Please upload a CSV file first.")
            return

        try:
            fixed_k = int(cluster_k) if int(cluster_k) >= 2 else None
            uploaded_file.seek(0)

            with st.spinner("Running analysis, please wait..."):
                results = backend.run_full_analysis(
                    uploaded_file,
                    fixed_k=fixed_k,
                    min_support=float(min_support),
                    min_confidence=float(min_confidence)
                )

            st.session_state["results"] = results

        except pd.errors.ParserError:
            st.error("Invalid CSV format. Please upload a valid CSV file.")

        except ValueError as error:
            st.error(str(error))

        except Exception as error:
            st.error(f"Unexpected error while running analysis: {error}")

    results = st.session_state.get("results")

    if results is None:
        st.info("Upload a CSV file in the sidebar and click Run Analysis to start.")
        return

    tabs = st.tabs([
        "Overview",
        "GMM Clustering",
        "t-SNE",
        "Association Rules",
        "Anomalies",
        "Insights"
    ])

    with tabs[0]:
        show_overview(results)

    with tabs[1]:
        show_clustering(results)

    with tabs[2]:
        show_tsne(results)

    with tabs[3]:
        show_association_rules(results)

    with tabs[4]:
        show_anomalies(results)

    with tabs[5]:
        show_insights(results)

    st.markdown("---")
    st.caption("Smart Shopping Behavior Analyzer | Data Mining Project")


if __name__ == "__main__":
    main()