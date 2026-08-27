# Strategic Planning & Data Preprocessing in Logistics

##  Project Overview
This project simulates an enterprise-level last-mile logistics optimization framework. It establishes dynamic routing strategies, spatial micro-hub clustering, and machine learning ETA predictions while enforcing a strict data collection, cleaning, and preprocessing pipeline for telemetry datasets.

##  Key Performance Indicators (KPIs)
* **On-Time Delivery Rate (OTD):** Target $\ge 95\%$
* **Cost Per Delivery (CPD):** Target 12% reduction
* **Average Route Idle & Transit Time:** Target 15% reduction

##  Project Structure
* `logistics_analysis.py`: Contains data preprocessing, cleaning, K-Means geospatial clustering, and Random Forest predictive models.
* `requirements.txt`: Python libraries needed to run the analysis.
* `README.md`: Project documentation and process roadmap.

##  Data Preprocessing Pipeline (Week 2 Highlights)
1. **Deduplication:** Removes duplicate `order_id` records.
2. **Text Normalization:** Standardizes categorical text formatting.
3. **Geospatial Bounding:** Filters out invalid coordinate values.
4. **Grouped Median Imputation:** Handles missing transit durations without skewing distribution.
5. **IQR Outlier Removal:** Eliminates GPS signal noise and extreme delays.
6. **Feature Scaling & Encoding:** Applies Min-Max scaling and One-Hot encoding.

##  How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt