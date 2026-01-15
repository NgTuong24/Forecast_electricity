import pandas as pd
from helper import time_features


# feature columns
X_FEATURE_INPUT = [
    "hour","dayofweek","is_weekend","month",
    # "lag_24","rolling_24",
    "airTemperature", "dewTemperature", "windSpeed",  # weather
    # "temp_lag_1h","dewTemperature_lag_1h", "windSpeed_lag_1h",
    # "sqft", 
    "sqm", 
    "primaryspaceusage", "site_id", "building_id",
    # 'chilled_delta', 'hot_delta',
    "Chilledwater", "Hotwater",
    'chilled_per_sqm', 'hot_per_sqm',
    'chilled_ratio_e', 'hot_ratio_e',
    # 'chilled_ratio_e_lag1', 'hot_ratio_e_lag1',
    'thermal_balance',
    'thermal_load', 'thermal_load_per_sqm'
]

def preprocess(raw_input: dict) -> pd.DataFrame:
    tf = time_features(raw_input["time"])
    sqm = raw_input.get("sqm", 0.0)
    Chilledwater = raw_input.get("Chilledwater", 0.0)
    Hotwater = raw_input.get("Hotwater", 0.0)
    electricity = raw_input.get("electricity", 0.0)
    
    feat = {
        "hour": tf["hour"],
        "dayofweek": tf["dayofweek"],
        "is_weekend": tf["is_weekend"],
        "month": tf["month"],
        
        "airTemperature": raw_input.get("air_temperature", 0.0),
        "dewTemperature": raw_input.get("dew_temperature", 0.0),
        "windSpeed": raw_input.get("wind_speed", 0.0),
        
        "sqm": sqm,
        
        "primaryspaceusage": raw_input.get("primaryspaceusage", "Unknown"),
        "site_id": raw_input.get("site_id", "Unknown"),
        "building_id": raw_input.get("building_code", "Unknown"),
        
        "Chilledwater": Chilledwater,
        "Hotwater": Hotwater,
        'chilled_per_sqm': float(Chilledwater/sqm+1e-6), 
        'hot_per_sqm': float(Hotwater/sqm+1e-6), 
        'chilled_ratio_e': float(Chilledwater/electricity+1e-6),
        'hot_ratio_e': float(Hotwater/electricity+1e-6),
        'thermal_balance': float((Chilledwater-Hotwater)/(Chilledwater+Hotwater+1e-6))*100,
        'thermal_load': float(Chilledwater+Hotwater),
        'thermal_load_per_sqm': float(Chilledwater+Hotwater)/sqm
    }

    return pd.DataFrame([[feat[col] for col in X_FEATURE_INPUT]],
                        columns=X_FEATURE_INPUT), tf["datetime"]
