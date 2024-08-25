import streamlit as st
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import requests

# Function to process uploaded data file
def process_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx'):
        return pd.read_excel(file)
    else:
        st.error("Unsupported file format")
        return None

# Streamlit layout
st.set_page_config(layout="wide")

# Sidebar for inputs
with st.sidebar:
    st.title("Query and Data Input")
    query = st.text_input("Enter your NLP query here:")
    data_file = st.file_uploader("Upload your data file", type=['csv', 'xlsx'])

# Main area layout
col1, col2 = st.columns([1, 3])

with col1:
    st.write("Operations")
    if st.button('Describe Data'):
        if data_file is not None:
            data = process_data(data_file)
            if isinstance(data, pd.DataFrame):
                st.write(data.describe())
                st.session_state['data'] = data.describe().to_dict()
            else:
                st.error("Unable to process data file.")
        else:
            st.error("Please upload a file first.")

    if st.button('Send'):
        if query:
            if 'data' in st.session_state:
                data = st.session_state['data']
                response = requests.post("http://localhost:5000/predict", json={"data": data, "query": query})
                
                if response.ok:
                    prediction = response.json().get("prediction")
                    python_code = extract_code(prediction)
                    st.session_state['python_code'] = python_code
                    st.code(python_code, language='python')
                else:
                    st.error("Error: Unable to get prediction from server.")
            else:
                st.error("Please describe the data first.")
        else:
            st.error("Please enter a query.")

with col2:
    st.write("Display Area")
    if 'python_code' in st.session_state:
        python_code = st.session_state['python_code']
        try:
            # Create a local variable dictionary to capture variables
            local_vars = {}
            exec(python_code, {'plt': plt, 'px': px, 'pd': pd, 'st': st}, local_vars)
        except Exception as e:
            st.error(f"Error executing the code: {e}")
