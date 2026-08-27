# Comprehensive Last-Mile Logistics Analytics & Dynamic Routing Framework

##  Project Overview
An end-to-end data science and operations research workspace simulating last-mile e-commerce delivery logistics. The project integrates dynamic fleet strategy, automated telemetry preprocessing, exploratory visual analytics, spatial clustering, and tuned ensemble predictive models to optimize transit duration forecasts and reduce Cost Per Delivery (CPD).

##  Key Performance Indicators (KPIs)
* **On-Time Delivery Rate (OTD):** Benchmark $\ge 95\%$
* **Cost Per Delivery (CPD):** 12% - 15% reduction target
* **Average Route Idle & Transit Time:** 15% reduction target

##  Weekly Technical Roadmap
* **Week 1: Strategic Planning & Architecture:** System setup, scenario definitions, KPI formulation, and initial pipeline planning.
* **Week 2: Data Preprocessing & Cleaning:** Deduplication, IQR outlier filtering, median imputation, and Min-Max feature scaling.
* **Week 3: Visual Exploratory Data Analysis (EDA):** Feature correlation analysis, delay distribution fitting (KDE), and traffic density box-plots.
* **Week 4: Machine Learning & Optimization:** Predictive modeling using Linear Regression, XGBoost/Gradient Boosting, and GridSearchCV-tuned Random Forest models ($R^2 = 0.9648$).

##  Required Libraries (`requirements.txt`)
```txt
pandas
numpy
scikit-learn
matplotlib
seaborn