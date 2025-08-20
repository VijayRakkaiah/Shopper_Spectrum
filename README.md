# 🛍️ Shopper Spectrum – Customer Segmentation & Product Recommendation

## 📌 Project Overview

**Shopper Spectrum** is a data science project focused on customer segmentation and product recommendation using the *Online Retail* dataset.

The project leverages **RFM (Recency, Frequency, Monetary)** analysis for customer profiling and **K-Means clustering** for segmentation. Additionally, it implements an **item-to-item recommender system** using cosine similarity to suggest similar products based on co-purchase patterns.

---

## 🖼️ Streamlit App Screenshots

### 1. Customer Recommendation Dashboard
![Customer Segmentation](.\assets\recommendation.png)

### 2. Product Segmentation Dashboard
![Product Recommendation](.\assets\segmentation.png)

---

## 🎯 Objectives

- Clean and preprocess the raw retail transaction data.
- Perform exploratory data analysis (EDA) to understand customer and product trends.
- Build RFM features to represent customer behavior.
- Cluster customers into meaningful segments (e.g., High-Value, Regular, Occasional, At-Risk).
- Develop an item-to-item recommender system for cross-selling.
- Save models/artifacts (`.pkl`) for reuse in deployment - streamlit

---

## 🗂️ Project Structure

```bash
Shopper_Spectrum.ipynb      # Main Jupyter Notebook  
online_retail.csv           # Dataset
scaler.pkl                  # StandardScaler object (saved)  
kmeans_model.pkl            # Trained K-Means model  
product_similarity.pkl      # Cosine similarity matrix  
product_names.pkl           # StockCode-to-Description lookup
app.py                      # Streamlit app  
README.md                   # Project documentation  
```
## 🛠️ Tech Stack

**Language:** Python 3

**Libraries Used:**

- **Data Handling** → `pandas`, `numpy`  
- **Visualization** → `matplotlib`, `seaborn`
- **Machine Learning** → `scikit-learn`  
- **Similarity** → `cosine_similarity` from `sklearn.metrics.pairwise`

---

## 📊 Key Steps

### 1. Data Cleaning

- Removed duplicates, missing `CustomerID`, and cancelled invoices (InvoiceNo starting with 'C').
- Removed invalid quantities/prices.
- Converted `InvoiceDate` to datetime format.

### 2. Exploratory Data Analysis (EDA)

- Analyzed top countries and products.
- Tracked monthly invoice trends.
- Explored distribution of invoice values and customer spend.

### 3. RFM Feature Engineering

- **Recency**: Days since last purchase.  
- **Frequency**: Number of unique invoices.  
- **Monetary**: Total spend.

Built an RFM table for each customer.

### 4. Customer Segmentation (K-Means)

- Standardized RFM features.
- Used Elbow method & Silhouette score to select `k=4`.
- Interpreted clusters into business-friendly labels:
  - High-Value  
  - Regular  
  - Occasional  
  - At-Risk
- Visualized clusters using PCA.

### 5. Product Recommendation (Cosine Similarity)

- Built a Customer × Product matrix using purchase quantities.
- Applied cosine similarity to find related products.
- Created a function `recommend_by_product_name()` to suggest top-N similar items.

### 6. Saving Models

- Exported trained models and similarity matrices as `.pkl` files for future deployment.

---

## 🚀 Future Improvements

- Add offline evaluation metrics for recommender (precision@k, recall@k).
- Explore association rule mining (Apriori / FP-Growth) for market-basket insights.
- Build a Streamlit app for interactive segmentation & recommendations.
- Apply alternative clustering methods (DBSCAN, Gaussian Mixture).

---

## 📚 Learnings

- How to apply RFM analysis for customer behavior profiling.
- Practical use of K-Means clustering and validation methods.
- Building a product recommender system with transaction data.
- Saving ML artifacts for deployment.

