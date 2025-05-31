import os.path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
from llm_pipeline import NLPPandasPipeline
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


class SensorRegistry:
    SENSOR_FILE =  "sensors.csv"

class WeatherData(BaseModel):
    timestamp: str
    sensor_id: int
    temperature: float
    humidity: float
    wind_speed: float

class WeatherDataRequest(BaseModel):
    data: List[WeatherData]

class QueryRequest(BaseModel):
    query: str

class Sensor(BaseModel):
    sensor_id: int
    location: str
# AIzaSyA0s1-XmiWc_ao8F106Qc6h9z0Eq2FWa4s
def get_pipeline():
    return NLPPandasPipeline(csv_path="data.csv", api_key="AIzaSyA0s1-XmiWc_ao8F106Qc6h9z0Eq2FWa4s")
@app.post("/weather-data")
async def add_weather_data(request: WeatherDataRequest):
    try:
        pipeline = get_pipeline()
        logging.info(f"Request body {request.data}")
        # new_data = pd.DataFrame([entry for entry in request.data])
        new_data = pd.DataFrame([entry.dict() for entry in request.data])
        if not os.path.exists(SensorRegistry.SENSOR_FILE):
            return {
                "status": "error",
                "message": "No sensors registered yet. Please register the sensor First."
            }
        sensors_df = pd.read_csv(SensorRegistry.SENSOR_FILE)

        registered_sensors = sensors_df['sensor_id'].tolist()

        logging.info(f"registered sensors are {registered_sensors}")
        sensor_id = request.data[0].sensor_id

        if  sensor_id not in registered_sensors:
            raise HTTPException(
                status_code=400,
                detail=f"The sensor ID {sensor_id} is not registered. Please register it first using /register-sensor."
            )
        pipeline.append_readings(new_data)
        return {"status": "success"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/query")
async def query_pipeline(request: QueryRequest):
    try:
        pipeline = get_pipeline()
        logging.info("The query from the UI is {}".format(request.query))
        # print(f"The query which is received from the UI  is {request.query}")

        response = pipeline.run(request.query)

        if isinstance(response, str):
            return {"response": response}
        logging.info(response)

        # return {"status": "success"}

        extracted_text = response.message.blocks[0].text
        return {"response": extracted_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Something went wrong. Please try again later.")


@app.post("/register_sensor")
async def registering_sensor(sensor: Sensor):
    try:
        if os.path.exists(SensorRegistry.SENSOR_FILE):
            sensors_df = pd.read_csv("sensors.csv")
        else:
            sensors_df = pd.DataFrame(columns=["sensor_id", "location"])
        print(sensors_df)

        if sensor.sensor_id in sensors_df["sensor_id"].values:
            present_location = sensors_df[sensors_df["sensor_id"] == sensor.sensor_id]["location"].values[0]
            return {
                "status": "already registered",
                "message": f"{sensor.sensor_id} is already present at location '{present_location}'."
            }
        df = pd.concat([sensors_df, pd.DataFrame([sensor.dict()])], ignore_index=True)

        df.to_csv(SensorRegistry.SENSOR_FILE, index=False)
        return {
            "status": "sensor registered",
            "message": f"Sensor ID {sensor.sensor_id} has been registered at location '{sensor.location}'."
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))