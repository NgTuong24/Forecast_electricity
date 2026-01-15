from helper import load_models
from preprocess import preprocess
from postprocess import postprocess
from encoder import LabelEncoder

class ElectricityForecaster:
    def __init__(
        self,
        model_dir: str,
        encode_path: str = None,
        forecast_horizon: int = 24
    ):
        self.encoder = LabelEncoder.load(encode_path)
        self.model_dir = model_dir
        self.horizon = forecast_horizon
        self.models = self.load_models()

    def load_models(self):
        return load_models(self.model_dir, self.horizon)

    def __call__(self, raw_input: dict):
        X, start_time = preprocess(raw_input)
        print(X)
        print("____"*5)
        X = self.encoder.transform(X)
        print(X)
        preds = {}
        for h, model in self.models.items():
            preds[h] = model.predict(X)[0]

        return postprocess(start_time, preds)

    
if __name__ == "__main__":
    MODEL_DIR = r"C:\Users\tuong\VSCode\Forecast_electricity\models_1578_csv"
    encode_path = r"C:\Users\tuong\VSCode\Forecast_electricity\models_1578_csv/categorical_encoder.pkl"
    default_input = {
        
        'sub_primaryspaceusage': 'Education', 
        'industry': None, 
        'subindustry': None, 
        'lat': 37.871903400000036, 
        'lng': -122.26072860000008, 
        'timezone': 'US/Pacific', 
        'heatingtype': None, 
        'yearbuilt': 1953, 
        'date_opened': None, 
        'numberoffloors': 5, 
        'occupants': None, 
        'energystarscore': None, 
        'eui': None, 
        'site_eui': None, 
        'source_eui': None, 
        'leed_level': None, 
        'rating': None, 
        'air_temperature': 16.11578947368421, 
        'cloud_coverage': None, 
        'dew_temperature': 13.3, 
        'precip_depth_1hr': 0.0, 
        'precip_depth_6hr': None, 
        'sea_lvl_pressure': 1020.7, 
        'wind_direction': 80.0, 
        'wind_speed': 2.131578947368421, 
        'id': 66, 
        'electricity': 98.25, 
        'hotwater': 0.0, 
        'chilledwater': 0.0, 
        'steam': 0.0, 
        'water': 0.0, 
        'irrigation': 0.0, 
        'solar': 0.0, 
        'gas': 0.0,

        'time': '2017-08-31T06:00:00',  
        'building_code': 'ggg', 
        'site_id': 'ff', 
        'sqm': 20402.2, 
        'sqft': 219608, 
        'primaryspaceusage': 'Education', 
        
        'Chilledwater': 4151.7687,
        'Hotwater': 10306.8675,
        # 'Hotwater': 105306.8675,
        # "lag_24": 99,
        # "rolling_24": 110.28203333333333,
        # "temp_lag_1h": 18.647368421052633,
        # "dewTemperature_lag_1h": 14.126315789473683,
        # "windSpeed_lag_1h": 2.263157894736842,
    } 

    forecaster = ElectricityForecaster(MODEL_DIR, encode_path, forecast_horizon=24)

    result = forecaster(default_input)
    
    import pandas as pd
    df = pd.DataFrame(result)
    df["time"] = pd.to_datetime(df["time"])

    print(df)