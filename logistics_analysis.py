import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 1. DATA CLEANING & PREPROCESSING PIPELINE
# ==========================================
def clean_logistics_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw logistics telemetry data by filtering coordinates,
    handling missing values, and removing outliers.
    """
    # Valid coordinates filter
    df = df[(df['latitude'].between(-90, 90)) & (df['longitude'].between(-180, 180))].copy()
    
    # Fill missing transit times with group median
    if 'transit_time_minutes' in df.columns and 'route_id' in df.columns:
        df['transit_time_minutes'] = df.groupby('route_id')['transit_time_minutes'].transform(
            lambda x: x.fillna(x.median())
        )
    
    # Remove extreme outliers in delivery duration using IQR
    if 'delivery_duration' in df.columns:
        Q1 = df['delivery_duration'].quantile(0.25)
        Q3 = df['delivery_duration'].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df['delivery_duration'] >= (Q1 - 1.5 * IQR)) & (df['delivery_duration'] <= (Q3 + 1.5 * IQR))]
    
    # Time-based features
    if 'dispatch_time' in df.columns:
        df['dispatch_time'] = pd.to_datetime(df['dispatch_time'])
        df['hour_of_day'] = df['dispatch_time'].dt.hour
        df['day_of_week'] = df['dispatch_time'].dt.dayofweek
        
    return df

# ==========================================
# 2. GEOSPATIAL CLUSTERING FOR MICRO-ZONES
# ==========================================
def generate_delivery_clusters(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """
    Groups delivery locations into optimal micro-zones using K-Means.
    """
    coords = df[['latitude', 'longitude']].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['delivery_zone_cluster'] = kmeans.fit_predict(coords)
    
    print("--- Micro-Hub Cluster Centroids ---")
    for idx, center in enumerate(kmeans.cluster_centers_):
        print(f"Zone {idx}: Lat {center[0]:.4f}, Lon {center[1]:.4f}")
        
    return df

# ==========================================
# 3. PREDICTIVE DURATION MODELING
# ==========================================
def train_delivery_time_model(df: pd.DataFrame):
    """
    Trains a Random Forest Regressor to predict total route duration.
    """
    features = ['package_count', 'total_weight_kg', 'distance_km', 'hour_of_day', 'day_of_week']
    target = 'total_route_time_minutes'
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    print("\n--- Model Evaluation ---")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, predictions)):.2f} minutes")
    print(f"R² Score: {r2_score(y_test, predictions):.4f}")
    
    return model

if __name__ == "__main__":
    print("Logistics Analytics Pipeline Loaded Successfully.")