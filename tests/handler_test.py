import unittest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import os
import pandas as pd

from handler import app, SensorRegistry


client = TestClient(app)



class TestAddWeatherDataEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.mock_sensor_file = "mock_sensors.csv"
        self.payload = {
            "data": [
                {
                    "timestamp": "2025-05-30 10:00:00",
                    "sensor_id": 1,
                    "temperature": 25.5,
                    "humidity": 60.0,
                    "wind_speed": 5.5
                }
            ]
        }

    @patch("handler.os.path.exists")
    @patch("handler.pd.read_csv")
    def test_sensor_valid_input(self, mock_read_csv, mock_exists):

        mock_exists.return_value = True
        mock_read_csv.return_value = pd.DataFrame({"sensor_id": [1]})
        import handler
        original_pipeline = handler.pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.append_readings = MagicMock()
        handler.pipeline = mock_pipeline


        response = client.post("/ingest-weather-data", json=self.payload)

        assert response.status_code == 200
        assert response.json()["status"] == "success"
        mock_pipeline.append_readings.assert_called_once()


if __name__ == "__main__":
    unittest.main()








