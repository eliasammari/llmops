import os
from flask import Flask, request, jsonify
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import GoogleGenerativeAI

# Set up API key and model
os.environ["GOOGLE_API_KEY"] = "AIzaSyB-q2NAqq5r2INU0yXat9VV1gEqtVRpidw"
gllm = GoogleGenerativeAI(model="gemini-pro", api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)

# Function to generate prompt template
def generate_prompt_template(data, query):
    template = """
    You are a teacher of computer science and you answer the following question:
    Here is the structure of the DataFrame you will be working with:

    {data}
    
    must start with ```python
    must end with ```
    must not contain any def function
    must contain only the final answer without all the previous text:
    Always use plotly if necessary or asked for a graph, else print only the value.
    Always start counting from the last record inside the dataset and not from the actual day.
    Always put the answer in a sentence.
      
    give me the python code to {query}
    Answer"""
    
    return template.format(data=data, query=query)

# Function to extract python code from response
def extract_code(response):
    try:
        start = response.find("```python") + len("```python")
        end = response.find("```", start)
        python_code = response[start:end].strip()
        return python_code
    except Exception as e:
        return f"Error extracting code: {str(e)}"

# Function to handle agent query
def handle_agent_query(data, query):
    try:
        memory = ConversationBufferMemory()
        prompt_template = generate_prompt_template(data, query)
        
        chain = ConversationChain(
            llm=gllm,
            memory=memory,
            verbose=True
        )
        
        response = chain.run(input=prompt_template)
        
        if not response:
            return "Error: No response from the model."
        
        return response
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json.get('data')
    query = request.json.get('query')
    response = handle_agent_query(data, query)
    return jsonify({'prediction': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
