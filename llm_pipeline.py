import pandas as pd
from llama_index.core import PromptTemplate
from llama_index.core.query_pipeline import QueryPipeline as QP, Link, InputComponent
from llama_index.experimental.query_engine.pandas import PandasInstructionParser
from llama_index.experimental.query_engine.pandas.output_parser import default_output_processor

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import Document
# api_key = "AIzaSyA0s1-XmiWc_ao8F106Qc6h9z0Eq2FWa4s"
# path = "D:\Downloads\project\data.csv"


class NoDataAvailableError(Exception):
    pass

class PandasParser(PandasInstructionParser):
    def __init__(self, df):
        super().__init__(df)

    def parse(self, text:str):
        print("The llm generated pandas Query is {}".format(text))
        result = default_output_processor(text, self.df)
        print("The result generated in the class is  {}".format(result))
        print('####################')
        print("The type of the result is {}".format(type(result)))
        if "nan" in result or "NaN" in result:
            raise NoDataAvailableError("Input is incorrect, Please try again with different input")

        return result





class NLPPandasPipeline:
    def __init__(self, csv_path: str, api_key: str):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.llm = GoogleGenAI(model="gemini-2.0-flash", api_key=api_key)
        self.retriever = self.build_retriever()

    def build_retriever(self):
        examples = [
            ("What is the average temperature for sensor 1 in the last 7 days?",
             "df[(df['sensor_id'] == 1) & (df['timestamp'] >= pd.Timestamp.now() - pd.Timedelta(days=7))]['temperature'].mean()"),
            ("What is the average humidity and wind speed for all sensors in the past month.",
             "df[df['timestamp'] >= pd.Timestamp.now() - pd.Timedelta(days=30)][['humidity', 'wind_speed']].mean()"),
            ("How many readings did sensor 3 report today?",
             "df[(df['sensor_id'] == 3) & (df['timestamp'].dt.date == pd.Timestamp.now().date())].shape[0]"),
            ("What is the maximum temperature recorded by sensor 2 this week.",
             "df[(df['sensor_id'] == 2) & (df['timestamp'] >= pd.Timestamp.now() - pd.Timedelta(days=7))]['temperature'].max()"),
            ("What is the average temperature per sensor in the last 24 hours?",
             "df[df['timestamp'] >= pd.Timestamp.now() - pd.Timedelta(hours=24)].groupby('sensor_id')['temperature'].mean()"),
            ("What is the temperature recorded by sensor 1? ",
             "df[df['sensor_id'] == 1].sort_values(by='timestamp', ascending=False)reset_index(drop=True).iloc[0]['temperature']"),
            ("What is the temperature recorded by sensor 2? ",
             "df[df['sensor_id'] == 2].sort_values(by='timestamp', ascending=False)reset_index(drop=True).iloc[0]['temperature']"),
            ("What is the average temperature recorded by sensor 1? ",
             "df[df['sensor_id'] == 1]['temperature'].mean()"),
            ("What is the average temperature and humidity recorded by sensor 1 and sensor 2 ?",
             "df[df['sensor_id'].isin([1, 2])][['temperature', 'humidity']].mean()"),
            ("Give me the average temperature and humidity for the sensor 1 and sensor 2 in last 30 days",
             "df[((df['sensor_id'] == 1)|(df['sensor_id'] == 2)) & (df['timestamp'] >= pd.Timestamp.now() - pd.Timedelta(days=30))][['temperature', 'humidity']].mean()"),
            ("what is the average temperature recorded on 2025-05-22",
             "df[df['timestamp'].dt.date == pd.to_datetime('2025-05-22').date()]['temperature'].mean()"),
            ("What is the average windspeed recorded on 22 May",
             "df[df['timestamp'].dt.day == 22]['wind_speed'].mean()")

        ]


        docs = [Document(text=f"Query: {q}\nAnswer: {a}") for q, a in examples]
        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)
        return index.as_retriever(similarity_top_k=3)

    def get_prompt(self, query_str: str) -> PromptTemplate:
        pandas_prompt_str = (
            "You are working with a pandas dataframe in Python.\n"
            "The name of the dataframe is `df`.\n"
            "This is the result of `print(df.head())`:\n"
            "{df_str}\n\n"
            "Here are some similar examples:\n"
            "{few_shot_examples}\n\n"
            "Follow these instructions:\n"
            "{instruction_str}\n"
            "Query: {query_str}\n\n"
            "Expression:"
        )
        instruction_str = (
            "1. Convert the query to executable Python code using Pandas.\n"
            "2. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
            "3. The code should represent a solution to the query.\n"
            "4. If the query refers to sensor data but does not specify any time range or date, "
                 "assume the user wants the **latest available value** based on the `timestamp` column.\n"
            "5. PRINT ONLY THE EXPRESSION.\n"
            "6. Do not quote the expression.\n"
        )
        few_shot_examples = "\n\n".join([node.text for node in self.retriever.retrieve(query_str)])
        print("The few shot examples are {}".format(few_shot_examples))
        return PromptTemplate(pandas_prompt_str).partial_format(
            instruction_str=instruction_str,
            df_str=self.df.head(5),
            few_shot_examples=few_shot_examples
        )

    def run(self, query_str: str):
        pandas_prompt = self.get_prompt(query_str)
        pandas_output_parser = PandasParser(self.df)
        print("######################")
        print("Pandas Output Parser output is {}".format(pandas_output_parser))
        response_synthesis_prompt = PromptTemplate(
            "Given an input question, synthesize a response from the query results.\n"
            "Query: {query_str}\n\n"
            "Pandas Instructions (optional):\n{pandas_instructions}\n\n"
            "Pandas Output: {pandas_output}\n\n"
            "Response:"
        )

        qp = QP(
            modules={
                "input": InputComponent(),
                "pandas_prompt": pandas_prompt,
                "llm1": self.llm,
                "pandas_output_parser": pandas_output_parser,
                "response_synthesis_prompt": response_synthesis_prompt,
                "llm2": self.llm,
            },
            # verbose=True,
        )

        qp.add_chain(["input", "pandas_prompt", "llm1", "pandas_output_parser"])
        # print(f"The output from llm1 is {llm1}")
        qp.add_links([
            Link("input", "response_synthesis_prompt", dest_key="query_str"),
            Link("llm1", "response_synthesis_prompt", dest_key="pandas_instructions"),
            Link("pandas_output_parser", "response_synthesis_prompt", dest_key="pandas_output"),
        ])
        qp.add_link("response_synthesis_prompt", "llm2")

        try:
            answer = qp.run(query_str = query_str)
            print("&&&&&&&&&&&&&&&&&&")
            print(answer)
            print(type(answer))
            return answer

        except NoDataAvailableError as e:
            print("The pipeline has been stopped ")

            return "For the provided date ranges the data is not available"





    def append_readings(self, new_data: pd.DataFrame):
        new_data['timestamp'] = pd.to_datetime(new_data['timestamp'])
        self.df = pd.concat([self.df, new_data], ignore_index=True)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df.to_csv(self.csv_path, index=False)

