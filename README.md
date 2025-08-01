# Shopper-Spectrum-Customer-Segmentation-and-Product-Recommendations-in-E-Commerce
A machine learning project that segments customers based on RFM (Recency, Frequency, Monetary) analysis and recommends products using collaborative filtering. Built with Python and Streamlit.
---

## 📌 Problem Statement

The e-commerce industry produces vast amounts of transaction data daily. This project helps analyze such data to:

- Identify customer segments using clustering techniques
- Recommend products using collaborative filtering
- Improve customer engagement and drive business growth

---

## 🎯 Objectives

- Segment customers using unsupervised learning (KMeans)
- Recommend similar products using item-based collaborative filtering
- Build an interactive web application using Streamlit

---

## 🗃️ Dataset Overview

- **Source**: Online Retail Dataset (2022–2023)
- **Download**: [Google Drive](https://drive.google.com/file/d/1rzRwxm_CJxcRzfoo9Ix37A2JTlMummY-/view?usp=sharing)

### Features:

| Column       | Description                      |
|--------------|----------------------------------|
| InvoiceNo    | Unique invoice number            |
| StockCode    | Product/item code                |
| Description  | Product name                     |
| Quantity     | Quantity of products purchased   |
| InvoiceDate  | Date and time of purchase        |
| UnitPrice    | Price per product                |
| CustomerID   | Unique customer identifier       |
| Country      | Customer’s country               |

---

## 🔍 Project Workflow

### 1. Data Preprocessing

- Remove missing `CustomerID`
- Exclude canceled invoices (`InvoiceNo` starting with 'C')
- Remove rows with non-positive `Quantity` or `UnitPrice`

### 2. Feature Engineering (RFM)

- **Recency**: Days since last purchase
- **Frequency**: Number of transactions
- **Monetary**: Total amount spent

### 3. Clustering

- Standardize RFM features
- Apply KMeans clustering
- Use Elbow and Silhouette Score to determine optimal clusters

### 4. Recommendation System

- Use cosine similarity on customer-product matrix
- Recommend top 5 similar products for any selected product

---

## 🖥️ Streamlit App Features

### ✅ Customer Segmentation

- Input: Recency, Frequency, Monetary values
- Output: Predicted cluster label (e.g., High-Value, At-Risk)
