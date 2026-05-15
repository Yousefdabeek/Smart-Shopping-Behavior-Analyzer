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

.explain-box {
    background-color: #1e293b;
    border: 1px solid #334155;
    padding: 16px;
    border-radius: 14px;
    margin-bottom: 18px;
    color: #dbeafe;
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
    st.sidebar.write("Upload your store sales CSV file and click Run Analysis.")

    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
    run_button = st.sidebar.button("Run Analysis", type="primary")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Simple Settings")

    customer_groups = st.sidebar.number_input(
        "Customer groups (0 = automatic)",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )

    min_support = st.sidebar.slider(
        "Minimum product pattern frequency",
        min_value=0.005,
        max_value=0.20,
        value=0.02,
        step=0.005
    )

    min_confidence = st.sidebar.slider(
        "Minimum pattern confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.30,
        step=0.05
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### How to Use")
    st.sidebar.markdown(
        "1. Upload sales CSV\n"
        "2. Click Run Analysis\n"
        "3. Read the results in each tab"
    )

    return uploaded_file, run_button, customer_groups, min_support, min_confidence


def show_header():
    st.title("🛒 Smart Shopping Behavior Analyzer")

    st.caption(
        "A simple dashboard that helps store owners understand customers, products, and unusual shopping behavior."
    )

    st.markdown("""
    <div class="hero-card">
    <h3>Understand your store data without needing data mining knowledge</h3>

    <p>This dashboard answers practical business questions:</p>

    <ul>
        <li>What types of customers do I have?</li>
        <li>Which products are usually bought together?</li>
        <li>Which customers behave unusually?</li>
        <li>What useful business insights can I get from my sales data?</li>
    </ul>

    <p><b>The system performs the data mining automatically in the background.</b></p>
    </div>
    """, unsafe_allow_html=True)


def show_overview(results):
    st.markdown('<div class="section-title">Store Overview</div>', unsafe_allow_html=True)

    clean_data = results["clean_data"]
    raw_data = results["raw_data"]

    st.markdown("""
    <div class="explain-box">
    This page gives a quick summary of your uploaded sales file.
    It shows how many sales rows, customers, invoices, products, and total revenue exist in the data.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Sales Rows", f"{len(clean_data):,}")
    col2.metric("Customers", f"{clean_data['CustomerID'].nunique():,}")
    col3.metric("Invoices", f"{clean_data['InvoiceNo'].nunique():,}")
    col4.metric("Products", f"{clean_data['Description'].nunique():,}")
    col5.metric("Revenue", f"${clean_data['TotalPrice'].sum():,.0f}")

    st.markdown('<div class="section-title">Sample of Your Data</div>', unsafe_allow_html=True)

    st.dataframe(raw_data.head(12), use_container_width=True, height=320)

    left, right = st.columns([2, 1])

    with left:
        st.markdown("### Columns Found")
        st.write(", ".join(raw_data.columns))

    with right:
        st.markdown("### Missing Values")
        missing_values = raw_data.isna().sum().reset_index()
        missing_values.columns = ["Column", "MissingCount"]
        st.dataframe(missing_values, use_container_width=True, height=220)


def show_customer_groups(results):
    st.markdown('<div class="section-title">Customer Groups</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="explain-box">
    Customers are automatically grouped based on shopping behavior:
    total spending, number of purchases, product variety, and purchase recency.
    Each group represents customers with similar behavior.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    col1.metric("Customer Groups Found", results["best_k"])

    if results["group_quality_score"] is not None:
        col2.metric("Group Quality Score", f"{results['group_quality_score']:.4f}")
        st.caption("Higher score means the customer groups are more clearly separated.")
    else:
        col2.metric("Group Quality Score", "Not available")

    customers = results["customers"]

    group_counts = (
        customers["CustomerGroup"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    group_counts.columns = ["CustomerGroup", "Customers"]

    fig = px.bar(
        group_counts,
        x="CustomerGroup",
        y="Customers",
        color="CustomerGroup",
        text="Customers",
        title="Number of Customers in Each Group"
    )

    fig.update_layout(
        showlegend=False,
        plot_bgcolor="#111827",
        paper_bgcolor="#111827",
        font_color="white"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Group Summary")

    st.dataframe(results["group_summary"], use_container_width=True, height=280)

    st.download_button(
        "Download Customer Groups CSV",
        data=results["customers"].to_csv(index=False),
        file_name="customer_groups.csv",
        mime="text/csv"
    )


def show_customer_map(results):
    st.markdown('<div class="section-title">Customer Map</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="explain-box">
    Each point is one customer. Customers that appear close to each other have similar shopping behavior.
    This map helps you visually understand customer similarity.
    </div>
    """, unsafe_allow_html=True)

    customer_map = results["customer_map"]

    fig = px.scatter(
        customer_map,
        x="MapX",
        y="MapY",
        color=customer_map["CustomerGroup"].astype(str),
        hover_data=["CustomerID"],
        title="Customer Similarity Map",
        labels={"color": "Customer Group"}
    )

    fig.update_layout(
        plot_bgcolor="#111827",
        paper_bgcolor="#111827",
        font_color="white"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        "Download Customer Map CSV",
        data=results["customer_map"].to_csv(index=False),
        file_name="customer_map.csv",
        mime="text/csv"
    )


def show_products_bought_together(results):
    st.markdown('<div class="section-title">Products Bought Together</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="explain-box">
    This section shows products that customers often buy together.
    Store owners can use this for product placement, bundles, discounts, or promotions.
    </div>
    """, unsafe_allow_html=True)

    rules = results["product_rules"]

    if rules.empty:
        st.warning("No strong product-pair pattern found. Try lowering the frequency or confidence settings.")
        return

    top_rule = rules.iloc[0]

    st.markdown(
        f"""
        <div class="hero-card">
        <b>Strongest Product Pattern</b><br><br>
        Customers who buy <b>{top_rule['ProductsBought']}</b>
        also often buy <b>{top_rule['AlsoBought']}</b>.<br><br>
        Pattern Frequency: {top_rule['Frequency']:.3f} |
        Confidence: {top_rule['Chance']:.3f} |
        Relationship Strength: {top_rule['RelationshipStrength']:.3f}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        "Confidence means how often the pattern happens. Relationship Strength means how strong the product relationship is compared to random chance."
    )

    st.dataframe(rules.head(20), use_container_width=True)

    st.download_button(
        "Download Product Patterns CSV",
        data=rules.to_csv(index=False),
        file_name="product_patterns.csv",
        mime="text/csv"
    )


def show_unusual_customers(results):
    st.markdown('<div class="section-title">Unusual Customers</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="explain-box">
    This section highlights customers whose shopping behavior is very different from others.
    Examples include unusually high spending, very low activity, or strange purchase patterns.
    </div>
    """, unsafe_allow_html=True)

    unusual_customers = results["unusual_customers"]

    st.metric("Unusual Customers Found", len(unusual_customers))

    if unusual_customers.empty:
        st.info("No unusual customers were detected.")
        return

    display_columns = [
        "CustomerID",
        "TotalSpent",
        "NumTransactions",
        "AvgTransactionValue",
        "RecencyDays",
        "UnusualScore"
    ]

    st.dataframe(
        unusual_customers[display_columns].sort_values("UnusualScore"),
        use_container_width=True,
        height=320
    )

    st.download_button(
        "Download Unusual Customers CSV",
        data=unusual_customers[display_columns].to_csv(index=False),
        file_name="unusual_customers.csv",
        mime="text/csv"
    )


def show_business_insights(results):
    st.markdown('<div class="section-title">Business Insights</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="explain-box">
    This section summarizes the most important findings in simple language.
    </div>
    """, unsafe_allow_html=True)

    for insight in results["business_insights"]:
        st.markdown(f"- {insight}")


def main():
    show_header()

    uploaded_file, run_button, customer_groups, min_support, min_confidence = sidebar_controls()

    if run_button:
        if uploaded_file is None:
            st.error("Please upload a CSV file first.")
            return

        try:
            fixed_k = int(customer_groups) if int(customer_groups) >= 2 else None
            uploaded_file.seek(0)

            with st.spinner("Analyzing your store data..."):
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
        "Customer Groups",
        "Customer Map",
        "Products Bought Together",
        "Unusual Customers",
        "Business Insights"
    ])

    with tabs[0]:
        show_overview(results)

    with tabs[1]:
        show_customer_groups(results)

    with tabs[2]:
        show_customer_map(results)

    with tabs[3]:
        show_products_bought_together(results)

    with tabs[4]:
        show_unusual_customers(results)

    with tabs[5]:
        show_business_insights(results)

    st.markdown("---")
    st.caption("Smart Shopping Behavior Analyzer | Store Analytics Dashboard")


if __name__ == "__main__":
    main()