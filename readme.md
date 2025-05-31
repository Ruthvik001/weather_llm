Sensor Weather Data Application
--- 
## Features
1) Register sensors based on the location
2) Send the weather data per sensor which contains parameters temperature, sensor_id, windspeed, humidity, timestamp
3) Query the weather data using natural language like "Give me the average temperature and humidity for sensor1 in the last month " 
---

## Requirements and Setup files
1) ** Download the requirements using 
    "pip install -r requirements.txt"
2) ** Run the fast API server using the command
  "uvicorn handler:app --host 0.0.0.0 --port 8002"
3) ** Run the streamlit using the command 
  "streamlit run ui.py"
---
## API and UI access

1) ** For registering the sensor_id use the fast api swagger which can be accessed from the web using 
   "http://localhost:8002/docs#/"
2) ** And streamlit ui is used for querying and can be accessed from the web using  "http://localhost:8501"

---

## Input format 
1) For registering sensor
    Endpoint is POST /register-sensor
    ~~~
    Body
    {
      "sensor_id": 1,
      "location": "Mumbai"
    }
2) Sending weather data for sensor ID 
    Endpoint is POST /ingest-weather-data
 Body
~~~
 {
   "data": [
     {
       "timestamp": "2025-05-31T10:25:30",
       "sensor_id": 1,
       "temperature": 30,
       "humidity": 45,
       "wind_speed": 12
     }
   ]
 }

~~~