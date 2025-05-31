import unittest
from unittest.mock import MagicMock, patch

from numpy.ma.testutils import assert_equal

import llm_pipeline

from llm_pipeline import NLPPandasPipeline
import pandas as pd

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "timestamp": ["2025-05-29 10:00:25", "2025-05-30 12:23:12"],
            "sensor_id": [1,2],
            "temperature": [10,20],
            "humidity": [20,21],
            "wind_speed": [25,26]
        })
        self.test_csv = "test.csv"
        self.df.to_csv(self.test_csv, index=False)


    @patch("llm_pipeline.GoogleGenAI")
    @patch("llm_pipeline.VectorStoreIndex.from_documents")
    def test_append_readings(self, mockllm, mock_index):
        mockllm = MagicMock()

        mock_index.as_retriever.return_value = MagicMock()

        nlp_pipeline = llm_pipeline.NLPPandasPipeline(self.test_csv, api_key="dummy")

        new_data = pd.DataFrame({
            "timestamp": ["2025-05-02 10:00:00"],
            "sensor_id": [2],
            "temperature": [30.0],
            "humidity": [65],
            "wind_speed": [10.0]
        })


        nlp_pipeline.append_readings(new_data)
        assert_equal(len(nlp_pipeline.df), 3)


if __name__ == "__main__":
    unittest.main()

