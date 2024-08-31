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

# Function to extract Python code from the model response
def extract_code(response):
    try:
        start = response.find("```python") + len("```python")
        end = response.find("```", start)
        python_code = response[start:end].strip()
        return python_code
    except Exception as e:
        return f"Error extracting code: {str(e)}"

# Streamlit layout configuration
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
                st.write(data.describe())  # Display summary for user
                st.session_state['data'] = data  # Store full dataframe
                st.session_state['columns'] = list(data.columns) 

                # Display a general plot of the DataFrame
                st.write("General Data Plot:")
                fig, ax = plt.subplots()
                data.plot(ax=ax)  # General plot using df.plot()
                st.pyplot(fig)

            else:
                st.error("Unable to process data file.")
        else:
            st.error("Please upload a file first.")

    if st.button('Send'):
        if query:
            if 'data' in st.session_state:
                data = st.session_state['data']
                try:
                    # Convert data to dictionary format for JSON
                    columns_str = str(st.session_state['columns'])
                    response = requests.post("http://api:5000/send_predict", json={"data": columns_str, "query": query})
                    
                    if response.ok:
                        prediction = response.json().get("prediction")

                        # Display the raw prediction before extracting code
                        st.write("Raw Prediction:")
                        st.write(prediction)

                        # Extract code after displaying raw prediction
                        python_code = extract_code(prediction)
                        st.session_state['python_code'] = python_code

                        # Display the extracted Python code
                        st.write("Extracted Python Code:")
                        st.code(python_code, language='python')

                    else:
                        st.error(f"Error: Unable to get prediction from server. Status code: {response.status_code}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Request failed: {e}")
            else:
                st.error("Please describe the data first.")
        else:
            st.error("Please enter a query.")

    # Add the new RAG button here
    if st.button('RAG'):
        if query:
            if 'data' in st.session_state:
                data = st.session_state['data']
                try:
                    # Convert data to dictionary format for JSON
                    columns_str = str(st.session_state['columns'])
                    response = requests.post("http://api:5000/rag_predict", json={"data": columns_str, "query": query, "rag": True})
                    
                    if response.ok:
                        python_code = response.json().get("prediction")
                        st.session_state['python_code'] = python_code
                        # Display the extracted Python code
                        st.write("Extracted Python Code:")
                        st.code(python_code, language='python')

                    else:
                        st.error(f"Error: Unable to get prediction from server. Status code: {response.status_code}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Request failed: {e}")
            else:
                st.error("Please describe the data first.")
        else:
            st.error("Please enter a query.")
with col2:
    st.write("Display Area")
    if 'python_code' in st.session_state:
        python_code = st.session_state['python_code']
        try:
            # Execute the generated Python code within a safe local environment
            local_vars = {}
            
            # Add Streamlit functions to the execution environment
            exec(python_code, {'df': data ,'plt': plt, 'px': px, 'pd': pd, 'st': st}, local_vars)
            
            # Optionally, extract variables from local_vars to display them
            # For example, if 'fig' is created by the executed code:
            if 'fig' in local_vars:
                st.plotly_chart(local_vars['fig'])
                
        except Exception as e:
            st.error(f"Error executing the code: {e}")

