import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set global seaborn visual theme
sns.set_theme(style="whitegrid")

# =========================================================
# WEEK 3 & 4: DATA SIMULATION PIPELINE
# =========================================================

def generate_logistics_dataset(n_samples: int = 5000) -> pd.DataFrame:
    """Generates a complete logistics dataset with non-linear traffic and volume features."""
    np.random.seed(42)
    
    distance = np.random.uniform(2.0, 45.0, n_samples)
    traffic_index = np.random.uniform(1.0, 10.0, n_samples)
    volume = np.random.uniform(0.5, 12.0, n_samples)
    weight = volume * 14.5 + np.random.normal(0, 3, n_samples)
    hour = np.random.randint(7, 21, n_samples)
    day = np.random.randint(0, 7, n_samples)
    
    # Non-linear traffic penalty threshold past index 6.0
    traffic_penalty = np.where(traffic_index > 6.0, (traffic_index ** 1.8) * 0.8, traffic_index * 1.5)
    
    # Calculate target transit time (minutes)
    base_transit = (distance * 1.4) + traffic_penalty + (volume * 0.9) + np.random.normal(0, 3, n_samples)
    promised_transit = (distance / 30.0) * 60.0
    delays = np.maximum(0, base_transit - promised_transit)
    fuel_cost = (distance * 0.45) + (traffic_index * 1.8) + (volume * 0.75) + np.random.normal(0, 2, n_samples)
    
    df = pd.DataFrame({
        'order_id': [f"ORD-{10000+i}" for i in range(n_samples)],
        'latitude': np.random.uniform(28.4, 28.7, n_samples),
        'longitude': np.random.uniform(77.0, 77.3, n_samples),
        'distance_km': np.round(distance, 2),
        'traffic_density_index': np.round(traffic_index, 1),
        'shipment_volume_m3': np.round(volume, 2),
        'package_weight_kg': np.round(weight, 2),
        'hour_of_day': hour,
        'day_of_week': day,
        'transit_time_minutes': np.round(base_transit, 2),
        'promised_transit_min': np.round(promised_transit, 2),
        'delay_minutes': np.round(delays, 2),
        'fuel_cost_usd': np.round(fuel_cost, 2),
        'vehicle_type': np.random.choice(['Van', 'Bike', 'Truck'], size=n_samples)
    })
    return df

# =========================================================
# WEEK 2: PREPROCESSING PIPELINE
# =========================================================

def clean_logistics_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Full data cleaning, deduplication, IQR outlier removal, and Min-Max scaling."""
    df_clean = df.copy()
    df_clean = df_clean.drop_duplicates(subset=['order_id'])
    df_clean['vehicle_type'] = df_clean['vehicle_type'].str.strip().str.capitalize()
    
    # Interquartile Range (IQR) Outlier Filtering
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
# WEEK 3: EDA AND VISUALIZATION
# =========================================================

def perform_eda_and_visualization(df: pd.DataFrame):
    """Executes exploratory analysis and displays visual charts."""
    print("=== SUMMARY STATISTICS (WEEK 3) ===")
    print(df[['distance_km', 'transit_time_minutes', 'delay_minutes', 'fuel_cost_usd']].describe().T)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Logistics Performance Analytics', fontsize=16, fontweight='bold')

    sns.histplot(df['delay_minutes'], kde=True, ax=axes[0, 0], color='#1f77b4', bins=30)
    axes[0, 0].set_title('Delivery Delay Distribution', fontsize=12)

    sns.regplot(data=df, x='distance_km', y='fuel_cost_usd', ax=axes[0, 1],
                scatter_kws={'alpha': 0.3, 'color': '#2ca02c'}, line_kws={'color': 'red'})
    axes[0, 1].set_title('Trip Distance vs. Fuel Cost ($)', fontsize=12)

    corr = df[['distance_km', 'shipment_volume_m3', 'traffic_density_index', 
                'transit_time_minutes', 'delay_minutes', 'fuel_cost_usd']].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1, 0], cbar=True)
    axes[1, 0].set_title('Feature Correlation Heatmap', fontsize=12)

    df['traffic_binned'] = pd.cut(df['traffic_density_index'], bins=5)
    sns.boxplot(data=df, x='traffic_binned', y='delay_minutes', ax=axes[1, 1], palette='Oranges')
    axes[1, 1].set_title('Traffic Density vs. Delay Severity', fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# =========================================================
# WEEK 1 & 4: CLUSTERING & PREDICTIVE MODELING PIPELINE
# =========================================================

def run_clustering_and_modeling(df: pd.DataFrame):
    """Executes K-Means clustering and comparative Machine Learning modeling (Week 4)."""
    # 1. Spatial K-Means Clustering
    coords = df[['latitude', 'longitude']].values
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['delivery_zone_cluster'] = kmeans.fit_predict(coords)
    
    # 2. Machine Learning Predictive Modeling
    features = ['distance_km', 'traffic_density_index', 'shipment_volume_m3', 'package_weight_kg', 'hour_of_day', 'day_of_week']
    target = 'transit_time_minutes'
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'Random Forest (Default)': RandomForestRegressor(random_state=42)
    }
    
    print("\n=== MODEL PERFORMANCE COMPARISON (WEEK 4) ===")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        print(f"{name:25s} | MAE: {mae:.2f} min | RMSE: {rmse:.2f} min | R²: {r2:.4f}")
        
    # Hyperparameter Tuning for Random Forest
    print("\n=== TUNING BEST MODEL (RANDOM FOREST) ===")
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5]
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=cv, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_rf = grid.best_estimator_
    best_preds = best_rf.predict(X_test)
    
    print(f"Optimal Hyperparameters: {grid.best_params_}")
    print(f"Tuned RF Performance | MAE: {mean_absolute_error(y_test, best_preds):.2f} min | RMSE: {np.sqrt(mean_squared_error(y_test, best_preds)):.2f} min | R²: {r2_score(y_test, best_preds):.4f}")

if __name__ == "__main__":
    print("Step 1: Simulating Logistics Dataset...")
    df_raw = generate_logistics_dataset()
    
    print("\nStep 2: Executing Week 3 Visual EDA...")
    perform_eda_and_visualization(df_raw)
    
    print("\nStep 3: Executing Week 2 Data Cleaning & Preprocessing...")
    df_clean = clean_logistics_pipeline(df_raw)
    
    print("\nStep 4: Running Week 1 & 4 Clustering and ML Predictive Pipeline...")
    run_clustering_and_modeling(df_clean)