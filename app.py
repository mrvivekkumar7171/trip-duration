# To Run : python app.py
from src.features.feature_definitions import feature_build
from pydantic import BaseModel
from fastapi import FastAPI
from joblib import load
import pandas as pd
from src.logger import infologger

app = FastAPI()

class PredictionInput(BaseModel):
    """Define the input parameters required for making predictions"""
    vendor_id: float
    pickup_datetime: float
    passenger_count: float
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    store_and_fwd_flag: float

try:
    """Load the pre-trained model using joblib
    """
    model_path = "models/model.joblib"
    model = load(model_path)
except Exception as e:
    infologger.info(f'Model loading has been failed with error : {e}')
else:
    infologger.info('Model loaded successfully')

@app.get("/")
def home():
    return "Working fine"

@app.post("/predict")
def predict(input_data: PredictionInput):
    """
        Extract features from input_data and make predictions using the loaded model.
        Convert Pydantic input directly into a dictionary then into a 1-row pandas DataFrame.
    """

    try:
        """converting recieved input pydantic class to dataframe
        """
        # features = {
        #         'vendor_id': input_data.vendor_id,
        #         'pickup_datetime': input_data.pickup_datetime,
        #         'passenger_count': input_data.passenger_count,
        #         'pickup_longitude': input_data.pickup_longitude,
        #         'pickup_latitude': input_data.pickup_latitude,
        #         'dropoff_longitude': input_data.dropoff_longitude,
        #         'dropoff_latitude': input_data.dropoff_latitude,
        #         'store_and_fwd_flag': input_data.store_and_fwd_flag
        # }
        # features = pd.DataFrame(features, index=[0])

        features = input_data.model_dump()
        features = pd.DataFrame([features])
        return features
    except Exception as e:
        infologger.info(f'Dataframe creation from input dictionary has been failed with error : {e}') 

    features = feature_build(features, 'prod')
    
    try:
        """function to make prediction and returns a predicted output"""
        prediction = model.predict(features)[0].item()
    except Exception as e:
         infologger.info(f'Prediction has been failed because of error : {e}')
    else:
        return {"prediction": prediction}

if __name__ == "__main__":
    import uvicorn

    # uvicorn.run(app, host="127.0.0.1", port=8080)
    uvicorn.run(app, host="0.0.0.0", port=8080)