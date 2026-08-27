import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# =========================================================
# WEEK 2: DATA COLLECTION, CLEANING & PREPROCESSING PIPELINE
# =========================================================

def load_and_inspect_data(file_path: str) -> pd.DataFrame:
    """Loads raw logistics telemetry data and prints initial diagnostics."""
    df = pd.read_csv(file_path)
    print(f"--- Raw Dataset Diagnostics ---")
    print(f"Initial Shape: {df.shape}")
    print("Missing Values:\n", df.isnull().sum())
    return df

def clean_logistics_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Full data cleaning, imputation, outlier removal, and encoding pipeline."""
    df_clean = df.copy()
    
    # 1. Deduplication
    if 'order_id' in df_clean.columns:
        df_clean = df_clean.drop_duplicates(subset=['order_id'])
    
    # 2. Text Normalization
    if 'vehicle_type' in df_clean.columns:
        df_clean['vehicle_type'] = df_clean['vehicle_type'].str.strip().str.capitalize()
    
    # 3. Geospatial Bounding Box Validation
    if 'latitude' in df_clean.columns and 'longitude' in df_clean.columns:
        valid_coords = (df_clean['latitude'].between(-90, 90)) & (df_clean['longitude'].between(-180, 180))
        df_clean = df_clean[valid_coords]
    
    # 4. Grouped Median Imputation for Missing Transit Times
    if 'transit_time_minutes' in df_clean.columns and 'vehicle_type' in df_clean.columns:
        df_clean['transit_time_minutes'] = df_clean.groupby('vehicle_type')['transit_time_minutes'].transform(
            lambda x: x.fillna(x.median())
        )
    
    # Drop rows missing critical timestamps or essential location metrics
    drop_cols = [c for c in ['dispatch_timestamp', 'package_weight_kg'] if c in df_clean.columns]
    if drop_cols:
        df_clean = df_clean.dropna(subset=drop_cols)
    
    # 5. Outlier Removal using Interquartile Range (IQR)
    if 'transit_time_minutes' in df_clean.columns:
        Q1 = df_clean['transit_time_minutes'].quantile(0.25)
        Q3 = df_clean['transit_time_minutes'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean['transit_time_minutes'] >= lower_bound) & 
                            (df_clean['transit_time_minutes'] <= upper_bound)]
    
    # 6. Feature Normalization (Min-Max Scaling)
    scale_cols = [c for c in ['package_weight_kg', 'transit_time_minutes'] if c in df_clean.columns]
    if scale_cols:
        scaler = MinMaxScaler()
        scaled_names = [f"{col}_scaled" for col in scale_cols]
        df_clean[scaled_names] = scaler.fit_transform(df_clean[scale_cols])
    
    # 7. One-Hot Encoding for Categorical Features
    if 'vehicle_type' in df_clean.columns:
        df_clean = pd.get_dummies(df_clean, columns=['vehicle_type'], prefix='veh', drop_first=False)
    
    print(f"\nCleaned Dataset Shape: {df_clean.shape}")
    return df_clean

# =========================================================
# WEEK 1: GEOSPATIAL CLUSTERING & PREDICTIVE MODELING
# =========================================================

def generate_delivery_clusters(df: pd.DataFrame, n_clusters: int = 4) -> pd.DataFrame:
    """Groups delivery locations into optimal micro-zones using K-Means."""
    coords = df[['latitude', 'longitude']].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['delivery_zone_cluster'] = kmeans.fit_predict(coords)
    
    print("\n--- Micro-Hub Cluster Centroids ---")
    for idx, center in enumerate(kmeans.cluster_centers_):
        print(f"Zone {idx}: Lat {center[0]:.4f}, Lon {center[1]:.4f}")
        
    return df

def train_delivery_time_model(df: pd.DataFrame):
    """Trains a Random Forest Regressor to predict total route duration."""
    features = ['package_weight_kg', 'hour_of_day', 'day_of_week']
    # Select available features dynamically
    features = [f for f in features if f in df.columns]
    target = 'transit_time_minutes'
    
    if target in df.columns and len(features) > 0:
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
    else:
        print("\nModel training skipped: Required feature/target columns missing.")
        return None

if __name__ == "__main__":
    print("Logistics Analytics Pipeline Loaded Successfully.")