# Smart Shopping Behavior Analyzer

Professional, presentation-friendly Streamlit dashboard for unsupervised shopping analytics.

## Project Structure

```text
project/
│
├── data/
│   └── smart_shopping_2000_rows.csv
├── backend.py
├── app.py
└── outputs/
```

## Methods Included

- K-Means Clustering
- PCA (2D)
- Association Rule Mining (Apriori)
- Anomaly Detection (Isolation Forest)

## Required CSV Columns

- `CustomerID`
- `InvoiceNo`
- `Description`
- `Quantity`
- `UnitPrice`
- `InvoiceDate`

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Use `data/smart_shopping_2000_rows.csv` as a demo dataset.
# Smart Shopping Behavior Analyzer

A clean and presentation-ready Streamlit dashboard for unsupervised shopping behavior analysis.

## Features

- K-Means clustering (with silhouette-based `k` selection)
- PCA 2D visualization
- Association rule mining (Apriori)
- Anomaly detection (Isolation Forest)
- Plain-English insights for fast interpretation

## Required CSV Columns

- `CustomerID`
- `InvoiceNo`
- `Description`
- `Quantity`
- `UnitPrice`
- `InvoiceDate`

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Dashboard Tabs

1. Overview
2. Clustering
3. PCA
4. Association Rules
5. Anomalies
6. Insights
# Smart Shopping Behavior Analyzer

This is a beginner-friendly data mining project that analyzes shopping transactions using unsupervised learning techniques.

## Techniques Used

1. **K-Means Clustering**
   - Groups customers by shopping behavior
   - Uses feature scaling
   - Selects the best K using silhouette score

2. **PCA (Dimensionality Reduction)**
   - Reduces customer features to 2 principal components
   - Helps visualize customer groups
   - Shows explained variance ratio

3. **Association Rule Mining (Apriori)**
   - Finds product relationships in baskets
   - Uses support, confidence, and lift

4. **Anomaly Detection (Isolation Forest)**
   - Detects unusual customer behavior

## Required Dataset Columns

Your CSV file must contain:

- `CustomerID`
- `InvoiceNo`
- `Description`
- `Quantity`
- `UnitPrice`
- `InvoiceDate`

## Project Files

- `smart_shopping_backend.py` -> data loading, preprocessing, modeling, and insights
- `app.py` -> simple Streamlit interface
- `requirements.txt` -> required libraries

## Installation

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

## Workflow

1. Upload a CSV file (or provide a local path).
2. Click **Run Analysis**.
3. View:
   - Cluster summary
   - PCA scatter plot
   - Association rules
   - Anomaly results
   - Plain-English insights

## Example Insights

- "Cluster 1 contains the highest-spending customers."
- "Customers buying Bread often also buy Milk."
- "3 anomalous customers were detected."
# RetailSight — Unsupervised Retail Intelligence

**Course:** Data Mining (10672349), An-Najah National University — *Final Practical Project*  
**Supervisor:** Dr.-Ing. Ahmed Abualia  

**RetailSight** is an **interactive data mining** app on **shopping transactions**: upload a CSV and explore **hidden patterns** with **unsupervised** methods (clustering, PCA, Apriori association rules, Isolation Forest anomalies, and short text **insights**). Modeling is **>70% unsupervised**; there is no supervised classifier.

---

## What runs under the hood

| Step | Implementation (`smart_shopping_backend.py`) |
|------|-----------------------------------------------|
| Load | `pd.read_csv` in the UI; `load_data` for CLI |
| Preprocess | Drop bad rows, parse dates, `TotalPrice` |
| Features | Per-customer aggregates (`TotalSpent`, `NumTransactions`, …) |
| Clustering | `StandardScaler` + **K-Means**; *k* from best **silhouette** unless you fix *k* |
| PCA | 2 components on the same scaled matrix |
| Rules | Basket one-hot (`InvoiceNo` × item) + **Apriori** + **association_rules** |
| Anomalies | **Isolation Forest** on scaled features |
| Insights | Bullet strings from cluster stats, PCA variance, top rule, anomaly count |

---

## Course alignment (short)

- **Mandatory:** clustering (K-Means), dimensionality reduction (PCA).  
- **Electives:** association rule mining, anomaly detection (Isolation Forest).  
- **Unsupervised ≥70%:** all learning steps above are unsupervised; preprocessing only supports them.

---

## Dataset

CSV should include **CustomerID**, **InvoiceNo**, **Quantity**, **UnitPrice**, and **Description** or **StockCode** (for features and rules). **InvoiceDate** is optional (parsed if present).

Demo files: `sample_transactions.csv`, `sample_transactions_3500.csv`.

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
streamlit run app.py
```

CLI demo (optional):

```bash
python smart_shopping_backend.py
```

(Edit `file_path` inside the `if __name__ == "__main__"` block to point at your CSV.)

---

## Repository layout

| Path | Role |
|------|------|
| `smart_shopping_backend.py` | All mining logic in one module (same style as a course “backend” script) |
| `app.py` | Streamlit UI — `pd.read_csv` upload, calls `run_analysis_from_dataframe` |
| `requirements.txt` | Dependencies |
| `sample_transactions*.csv` | Demo data |

---

## Limitations

- **Small files:** few customers → unstable silhouette / clusters.  
- **K-Means** assumes roughly spherical groups.  
- **PCA** is linear (two components capture part of the variance only).  
- **Apriori** depends on support/confidence; sparse baskets may yield no rules.  
- **Isolation Forest** marks statistical outliers, not proven fraud.

---

## Team / AI / demo assets

Use the course template for team names, declare AI assistance if any, and add slides, screenshots, and a short demo video per instructor instructions.

## License

See `LICENSE`.
