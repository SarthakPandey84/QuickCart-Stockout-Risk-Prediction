# 🛒 QuickCart Stockout Risk Prediction

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)

> **An end-to-end Machine Learning project for predicting stockout risk across products and dark stores in a quick-commerce environment.**

---

## 📌 Project Overview

In quick-commerce, maintaining the right inventory level is critical. A stockout can lead to lost sales and poor customer experience, while excessive inventory can tie up capital and increase storage or spoilage risk.

**QuickCart Stockout Risk Prediction** uses historical inventory, sales, supplier, store, product, and event data to classify the daily stockout risk of each **Store–SKU combination** into three categories:

* 🟢 **Safe**
* 🟡 **At-Risk**
* 🔴 **Imminent**

The project focuses on an important operational question:

> **Which products are most likely to run out before their next replenishment arrives?**

---

## 🎯 Problem Statement

Quick-commerce inventory teams need to make frequent replenishment decisions across multiple stores and products.

The challenge is to identify products that may stock out before the next supplier delivery while avoiding unnecessary over-ordering.

The project formulates this as a **3-class supervised classification problem**.

### Target Classes

| Class           | Definition                                            | Distribution |
| --------------- | ----------------------------------------------------- | -----------: |
| 🟢 **Safe**     | Stock cover exceeds replenishment wait by 3+ days     |       65.42% |
| 🟡 **At-Risk**  | Stock cover is within 3 days of replenishment wait    |       24.01% |
| 🔴 **Imminent** | Stock is expected to run out before the next delivery |       10.57% |

Because the **Imminent** class is the minority class and represents the most operationally critical situation, model evaluation focuses particularly on its recall.

---

## 📊 Dataset

The project uses five related datasets representing stores, products, suppliers, events, and daily inventory activity.

| Dataset                    | Records | Description                       |
| -------------------------- | ------: | --------------------------------- |
| `dim_stores.csv`           |      12 | Store information                 |
| `dim_skus.csv`             |      60 | SKU/product information           |
| `dim_suppliers.csv`        |      15 | Supplier information              |
| `dim_events.csv`           |      30 | Festival and promotional events   |
| `fact_inventory_daily.csv` |  21,600 | Daily Store–SKU inventory records |

The fact table follows the grain:

```text
Store × SKU × Date
```

giving:

```text
12 Stores × 60 SKUs × 30 Days = 21,600 Records
```

### Dataset Period

**October 1, 2026 – October 30, 2026**

The dataset also contains event information such as **Diwali Week** and a **Weekend Flash Sale**, allowing the model to account for changes in demand associated with known events.

---

## 🧹 Data Preparation

Before modeling, the data is cleaned, integrated, and transformed into model-ready features.

Key preprocessing steps include:

* Handling missing supplier reliability values
* Standardizing relevant categorical values
* Joining dimension tables with daily inventory data
* Creating historical sales features
* Generating inventory and replenishment indicators
* Encoding categorical variables
* Scaling numerical features where required

Missing supplier reliability values are handled using **category-level median imputation**, following the project specification.

---

## ⚙️ Feature Engineering

Feature engineering combines inventory position, historical demand, supplier behavior, and event information.

### 📦 Inventory Features

Examples include:

* `reorder_gap`
* `days_of_cover`
* `days_of_cover_ratio`
* closing stock
* reorder information

### 📈 Demand Features

Historical demand is represented using lagged and rolling sales features, including:

* 3-day rolling sales average
* 7-day rolling sales average
* previous sales/inventory information

Rolling features are shifted using `.shift()` so that future observations are not used when constructing historical features.

### 🚚 Supplier Features

Supplier-related features include:

* Expected lead time
* Actual lead time where available
* Supplier reliability
* Cleaned supplier reliability

### 📅 Temporal & Event Features

The model also incorporates:

* Day of month
* Festival indicators
* Days since festival start
* Promotional/event information

These features help capture demand changes associated with known events.

---

## 🔎 Dataset Insights

The exploratory analysis reveals several relationships relevant to stockout risk.

### Festival Periods

The observed Imminent rate increases during festival periods:

| Period       | Imminent Rate |
| ------------ | ------------: |
| Non-Festival |         9.51% |
| Festival     |        23.31% |

This corresponds to approximately **2.45×** the non-festival rate.

### Supplier Reliability

The observed Imminent rate also varies across supplier reliability groups:

| Supplier Reliability | Imminent Rate |
| -------------------- | ------------: |
| `< 0.75`             |        15.83% |
| `0.75–0.85`          |         6.30% |
| `>= 0.85`            |         3.82% |

### Perishability

| SKU Type       | Imminent Rate |
| -------------- | ------------: |
| Non-perishable |          9.3% |
| Perishable     |         12.8% |

These figures describe patterns present in the project dataset and are not intended as causal estimates.

---

## 🤖 Machine Learning Approach

The project evaluates three classification algorithms:

* **Multinomial Logistic Regression**
* **Random Forest**
* **Gradient Boosting**

The modeling workflow is structured using Scikit-Learn's preprocessing and pipeline components.

### ML Pipeline

```text
Raw Data
   │
   ▼
Data Cleaning
   │
   ▼
Table Integration
   │
   ▼
Feature Engineering
   │
   ▼
Temporal Train/Test Split
   │
   ▼
Preprocessing
   │
   ├── Numerical Features
   │      └── Scaling
   │
   └── Categorical Features
          └── Encoding
   │
   ▼
Model Training
   │
   ├── Logistic Regression
   ├── Random Forest
   └── Gradient Boosting
   │
   ▼
Model Evaluation
   │
   ▼
Business-Focused Model Selection
```

---

## ⏱️ Train/Test Split

A **time-based split** is used rather than a random row-level split.

```text
Training
Oct 1 ───────────── Oct 23

Testing
                         Oct 24 ─── Oct 30
```

This approach is appropriate because the same Store–SKU combinations appear across multiple dates.

Using a random row-level split could allow observations from the same Store–SKU pair to appear in both training and testing data, making evaluation less representative of a future prediction scenario.

---

## 📏 Evaluation Metrics

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* **Imminent Recall**

### Why Imminent Recall?

For this problem, failing to identify an actual imminent stockout can be more costly than generating a false warning.

Therefore, the key question is:

> **Of all products that were actually Imminent, how many did the model successfully identify?**

This makes **Imminent Recall** an important metric for model comparison.

---

## 📈 Model Performance

The current evaluation produces the following results:

| Model                   |   Accuracy | Imminent Recall |
| ----------------------- | ---------: | --------------: |
| **Logistic Regression** |     88.79% |      **91.74%** |
| Random Forest           |     89.38% |          60.13% |
| Gradient Boosting       | **94.96%** |          76.39% |

### Key Observation

The models demonstrate a clear trade-off between **overall accuracy** and **detection of the Imminent class**.

Gradient Boosting achieves the highest overall accuracy, while Logistic Regression achieves substantially higher recall for the Imminent class.

For the project's defined business objective, this makes **Imminent Recall** an important consideration alongside overall accuracy.

---

## 💡 What This Project Demonstrates

This project goes beyond simply training multiple classification models.

It demonstrates an end-to-end ML workflow involving:

<div align="center">

**Business Problem**

↓

**Data Integration**

↓

**Data Cleaning**

↓

**Feature Engineering**

↓

**Leakage-Aware Temporal Split**

↓

**Multiple ML Models**

↓

**Business-Focused Evaluation**

</div>

The main takeaway is that **model selection depends on the problem objective and the cost of different prediction errors—not only on overall accuracy.**

---

## 🛠️ Tech Stack

| Technology           | Purpose                         |
| -------------------- | ------------------------------- |
| **Python**           | Core programming language       |
| **Pandas**           | Data manipulation and analysis  |
| **NumPy**            | Numerical operations            |
| **Scikit-Learn**     | Preprocessing and ML models     |
| **Matplotlib**       | Data visualization              |
| **Jupyter Notebook** | Development and experimentation |

---

## 📁 Repository Structure

```text
QuickCart-Stockout-Risk-Prediction/
│
├── data/
│   ├── dim_events.csv
│   ├── dim_skus.csv
│   ├── dim_stores.csv
│   ├── dim_suppliers.csv
│   └── fact_inventory_daily.csv
│
├── docs/
│   └── QuickCart_Stockout_Risk_Project_Spec.pdf
│
├── src/
│   └── QuickCart_Sarthak.ipynb
│
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd QuickCart-Stockout-Risk-Prediction
```

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib jupyter
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook src/QuickCart_Sarthak.ipynb
```

Then run the notebook cells sequentially or use **Run All**.

---

## ⚠️ Limitations

This project uses a controlled 30-day dataset and is intended as an academic/portfolio ML project.

Current limitations include:

* Limited temporal coverage of 30 days
* No live inventory or sales integration
* No real-world monetary cost assigned to prediction errors
* No production deployment
* Limited evidence for long-term seasonal generalization
* Target construction is closely related to inventory-cover and replenishment logic

These factors should be considered before applying the approach directly to a production inventory system.

---

## 🔮 Future Scope

Possible extensions include:

* Integration with real-time inventory systems
* Longer historical datasets
* Seasonal demand modeling
* Dynamic supplier lead-time prediction
* Cost-sensitive classification
* Automated stockout alerts
* Inventory recommendation systems
* Model monitoring and periodic retraining
* API or dashboard deployment

---

## 👨💻 Author

**Sarthak Pandey**  
*Batch: Unlox DS August 2026*  
*Roll no.: DS17725*

(Data Science Minor Project)

---

## 📄 Project Specification

Developed using the **QuickCart Warehouse Inventory Stockout Risk — Supervised ML Classification Project Specification** provided by Unlox Academy.

---

⭐ **If you found the project interesting, feel free to explore the notebook and the complete ML pipeline.**
