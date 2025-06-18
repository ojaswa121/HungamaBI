import streamlit as st
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os

# Load the API key from .env file
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="AskCSV (Local)", layout="wide")

st.title("📊 AskCSV — LLM-Powered CSV Explorer")
st.markdown("Upload a CSV and ask questions about it using natural language.")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
query = st.text_input("Ask a question about the data")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Preview of CSV")
    st.dataframe(df.head())

    if query:
        with st.spinner("Thinking..."):
            try:
                agent = create_pandas_dataframe_agent(
                    OpenAI(temperature=0, openai_api_key=openai_key),
                    df,
                    verbose=False,
                    allow_dangerous_code=True  # ✅ Required to allow code execution
                )
                response = agent.run(query)
                st.success("Answer:")
                st.markdown(response)
            except Exception as e:
                st.error(f"Error: {str(e)}")
