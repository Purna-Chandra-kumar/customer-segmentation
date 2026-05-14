# Customer Segmentation using KMeans Clustering

## Overview

This project is a complete **Customer Segmentation System** built using **Machine Learning (KMeans Clustering)** and **Streamlit Dashboard Visualization**.

The system generates synthetic customer data, performs clustering using the KMeans++ algorithm, visualizes customer groups, and provides business insights through interactive charts and dashboards.

The project helps businesses identify different customer groups based on:

* Annual Income
* Spending Score
* Age
* Purchase Behavior

---

## Features

* Generate synthetic customer dataset (1,000 customers)
* Perform customer segmentation using KMeans++
* Interactive Streamlit dashboard
* PCA-based cluster visualization
* Radar profile analysis
* Feature discriminability analysis
* Income vs Spending scatter visualization
* Export processed data
* Automated business insights generation

---

## Project Structure

```bash
customer-segmentation/
├── app.py
├── src/
│   ├── generate_data.py
│   ├── segmentation.py
│   ├── visualize.py
│   └── insights.py
├── requirements.txt
├── run.bat
├── run.sh
└── README.md
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Streamlit

---

## Workflow

### 1. Data Generation

`generate_data.py`

* Creates synthetic customer records
* Generates attributes like age, income, and spending score

### 2. Customer Segmentation

`segmentation.py`

* Applies KMeans++ clustering
* Groups customers into meaningful segments

### 3. Visualization

`visualize.py`

* Generates all graphical outputs
* Performs PCA dimensionality reduction

### 4. Business Insights

`insights.py`

* Creates segment-wise business analysis
* Suggests marketing strategies for each cluster

---

## Generated Charts

### PCA Cluster Visualization

Displays customer clusters in reduced dimensions using PCA.

### Radar Segment Profiles

Shows behavioral patterns and segment DNA.

### Feature Importance Analysis

Identifies which features contribute most to segmentation.

### Income vs Spending Scatter Plot

Visualizes customer spending habits against annual income.

---

## Installation

### Step 1: Extract ZIP

```bash
cd C:\Users\ASUS\Downloads
Expand-Archive -Path "customer-segmentation.zip" -DestinationPath "."
cd customer-segmentation
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Project

### Run Complete Pipeline

```bash
python src/generate_data.py
python src/segmentation.py
python src/visualize.py
python src/insights.py
```

### Launch Streamlit Dashboard

```bash
streamlit run app.py
```

Open:

```bash
http://localhost:8501
```

---

## GitHub Push Commands

```bash
git init
git add .
git commit -m "Customer Segmentation Project - KMeans Clustering"
git remote add origin https://github.com/Purna-Chandra-kumar/customer-segmentation.git
git branch -M main
git push -u origin main
```

---

## Applications

* Customer behavior analysis
* Personalized marketing
* Business intelligence
* Sales optimization
* Customer retention strategies

---

## Future Enhancements

* Real-world dataset integration
* Deep learning-based segmentation
* Real-time analytics dashboard
* Cloud deployment
* Recommendation system integration

---

## Conclusion

This project demonstrates how Machine Learning can be used to identify customer groups and improve business decision-making. Using clustering and interactive visualizations, businesses can better understand customer behavior and design targeted marketing strategies.
