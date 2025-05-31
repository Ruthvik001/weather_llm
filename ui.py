import streamlit as st
import requests
from pyexpat.errors import messages

st.title("Weather Data Query Interface")

query = st.text_input("Enter your query:")

if st.button("Submit Query"):
    try:
        response = requests.post("http://localhost:8002/query", json={"query":query}, timeout = 15)
        if response.status_code == 200:
            data = response.json()
            print(response)
            st.success(data["response"])
        else:
            st.error(f"Error {response.status_code}, {response.reason}")

    except Exception as e:
        st.error("Please a try a different question. Process failed with excpetion  {} ".format(e))