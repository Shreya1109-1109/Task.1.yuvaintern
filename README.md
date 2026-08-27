# Comprehensive Logistics Analytics & Dynamic Routing

##  Project Overview
An end-to-end logistics analytics workspace simulating an e-commerce last-mile fulfillment network. Features strategic KPI frameworks, automated data preprocessing, exploratory visual analytics, spatial K-Means clustering, and Random Forest transit prediction models.

##  Key Performance Indicators (KPIs)
* **On-Time Delivery Rate (OTD):** Target $\ge 95\%$
* **Cost Per Delivery (CPD):** Target 12% reduction
* **Average Route Idle & Transit Time:** Target 15% reduction

##  Exploratory Analytics & Visualization (Week 3)
* **Delay Frequency:** Histograms with KDE to measure tail risks in late deliveries.
* **Cost Trends:** Regression plots analyzing travel distance vs. fuel consumption.
* **Correlation Analysis:** Heatmaps identifying traffic density as the primary bottleneck ($r > 0.78$).
* **Congestion Variance:** Box plots showing exponential delay increases past traffic index level 6.5.

##  Pipeline Roadmap
1. **Week 1:** Strategic Planning & Machine Learning Framework
2. **Week 2:** Data Cleaning, Imputation, IQR Outlier Removal & Min-Max Scaling
3. **Week 3:** EDA, Statistical Summaries & Seaborn/Matplotlib Visualizations

##  How to Run
```bash
pip install -r requirements.txt
python logistics_analysis.py