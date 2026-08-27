import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Global seaborn style set
sns.set_theme(style="whitegrid")

# =========================================================
# WEEK 3: DATA SIMULATION, EDA & VISUALIZATION PIPELINE
# =========================================================

def generate_logistics_data(n_samples: int = 5000) -> pd.DataFrame:
    """Simulates a detailed logistics operational dataset for Week 3 EDA."""
    np.random.seed(42)
    
    distance = np.random.uniform(2.0, 40.0, n_samples)
    traffic_index = np.random.uniform(1.0, 10.0, n_samples)
    volume = np.random.uniform(0.5, 10.0, n_samples)
    
    base_speed_kmh = 35.0 - (traffic_index * 2.2)
    actual_transit = (distance / base_speed_kmh) * 60.0 + np.random.normal(0, 5, n_samples)
    promised_transit = (distance / 30.0) * 60.0
    
    delays = np.maximum(0, actual_transit - promised_transit)
    fuel_cost = (distance * 0.45) + (traffic_index * 1.8) + (volume * 0.75) + np.random.normal(0, 2, n_samples)
    
    df = pd.DataFrame({
        'order_id': [f"ORD-{10000+i}" for i in range(n_samples)],
        'latitude': np.random.uniform(28.4, 28.7, n_samples),
        'longitude': np.random.uniform(77.0, 77.3, n_samples),
        'distance_km': np.round(distance, 2),
        'shipment_volume_m3': np.round(volume, 2),
        'traffic_density_index': np.round(traffic_index, 1),
        'transit_time_minutes': np.round(actual_transit, 2),
        'promised_transit_min': np.round(promised_transit, 2),
        'delay_minutes': np.round(delays, 2),
        'fuel_cost_usd': np.round(fuel_cost, 2),
        'package_weight_kg': np.round(volume * 15.0 + np.random.normal(0, 2, n_samples), 2),
        'vehicle_type': np.random.choice(['Van', 'Bike', 'Truck'], size=n_samples)
    })
    return df

def perform_eda_and_visualization(df: pd.DataFrame):
    """Executes exploratory analysis and displays visual charts."""
    print("=== SUMMARY STATISTICS ===")
    print(df[['distance_km', 'transit_time_minutes', 'delay_minutes', 'fuel_cost_usd']].describe().T)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Logistics Performance Analytics (Week 3)', fontsize=16, fontweight='bold')

    # 1. Delay Distribution
    sns.histplot(df['delay_minutes'], kde=True, ax=axes[0, 0], color='#1f77b4', bins=30)
    axes[0, 0].set_title('Delivery Delay Distribution', fontsize=12)

    # 2. Distance vs Fuel Cost
    sns.regplot(data=df, x='distance_km', y='fuel_cost_usd', ax=axes[0, 1],
                scatter_kws={'alpha': 0.3, 'color': '#2ca02c'}, line_kws={'color': 'red'})
    axes[0, 1].set_title('Trip Distance vs. Fuel Cost ($)', fontsize=12)

    # 3. Correlation Matrix
    corr = df[['distance_km', 'shipment_volume_m3', 'traffic_density_index', 
                'transit_time_minutes', 'delay_minutes', 'fuel_cost_usd']].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1, 0], cbar=True)
    axes[1, 0].set_title('Feature Correlation Heatmap', fontsize=12)

    # 4. Traffic vs Delay Severity
    df['traffic_binned'] = pd.cut(df['traffic_density_index'], bins=5)
    sns.boxplot(data=df, x='traffic_binned', y='delay_minutes', ax=axes[1, 1], palette='Oranges')
    axes[1, 1].set_title('Traffic Density vs. Delay Severity', fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# =========================================================
# WEEK 2: PREPROCESSING PIPELINE
# =========================================================

def clean_logistics_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Full data cleaning and preprocessing pipeline."""
    df_clean = df.copy()
    df_clean = df_clean.drop_duplicates(subset=['order_id'])
    df_clean['vehicle_type'] = df_clean['vehicle_type'].str.strip().str.capitalize()
    
    # Remove outliers using IQR
    Q1 = df_clean['transit_time_minutes'].quantile(0.25)
    Q3 = df_clean['transit_time_minutes'].quantile(0.75)
    IQR = Q3 - Q1
    df_clean = df_clean[(df_clean['transit_time_minutes'] >= (Q1 - 1.5 * IQR)) & 
                        (df_clean['transit_time_minutes'] <= (Q3 + 1.5 * IQR))]
    
    scaler = MinMaxScaler()
    df_clean[['weight_scaled', 'transit_scaled']] = scaler.fit_transform(
        df_clean[['package_weight_kg', 'transit_time_minutes']]
    )
    return df_clean

# =========================================================
# WEEK 1: CLUSTERING & MODELING
# =========================================================

def run_clustering_and_modeling(df: pd.DataFrame):
    """Executes K-Means clustering and Random Forest regression."""
    coords = df[['latitude', 'longitude']].values
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['delivery_zone_cluster'] = kmeans.fit_predict(coords)
    
    features = ['package_weight_kg', 'distance_km', 'traffic_density_index']
    X = df[features]
    y = df['transit_time_minutes']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    print("\n--- Predictive Model Evaluation ---")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, predictions)):.2f} minutes")
    print(f"R² Score: {r2_score(y_test, predictions):.4f}")

if __name__ == "__main__":
    print("Generating Logistics Data...")
    df_raw = generate_logistics_data()
    
    print("\nExecuting Week 3 EDA and Visualizations...")
    perform_eda_and_visualization(df_raw)
    
    print("\nExecuting Week 2 Preprocessing...")
    df_clean = clean_logistics_pipeline(df_raw)
    
    print("\nExecuting Week 1 Analytics & ML Modeling...")
    run_clustering_and_modeling(df_clean)