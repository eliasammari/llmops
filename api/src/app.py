import os
from flask import Flask, request, jsonify
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import GoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore  # Updated import
import pinecone as pc

from pinecone import Pinecone
from langchain.prompts import PromptTemplate
from langsmith import Client

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_PROJECT"] = "Live-evaluation"
os.environ["PINECONE_API_KEY"] = "396738cd-4531-45b0-b166-39ec4b61735d"

client=Client()

#   AIzaSyAauQdTvqz0ynb15ZBL6lrhwzmpz5kkEh8
#   AIzaSyDpN7nVRpfQ16DgIHMhA0py2d7hFcLq8WM
#   AIzaSyB-q2NAqq5r2INU0yXat9VV1gEqtVRpidw
# Set up API key and model
os.environ["GOOGLE_API_KEY"] = "AIzaSyB-q2NAqq5r2INU0yXat9VV1gEqtVRpidw"
gllm = GoogleGenerativeAI(model="gemini-pro", api_key=os.getenv("GOOGLE_API_KEY"))

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# Configuration Pinecone
api_key = "396738cd-4531-45b0-b166-39ec4b61735d"
pc = Pinecone(api_key=api_key)

index_name = "plotly-md-files-index"
vectorestore = PineconeVectorStore(index_name=index_name, embedding=embeddings)


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
    must contain only the final answer without all the previous text.
    Always use plotly if necessary or asked for a graph, else print only the value.
    Always use fig.
    Always start counting from the last record inside the dataset and not from the actual day.
    Always put the answer in a sentence.
      
    give me the python code to {query}  based on the provided data in the prompt.
    Answer"""
    
    return template.format(data=data, query=query)

def def_rag_prompt_template(data):
    prompte_template = """
    You are a teacher of computer science, and you will answer the following question:
        
    Data provided:

    {data}
        
    Instructions:
    - must start with ```python
    - must end with ```
    - must not contain any def function
    - must contain only the final answer without all the previous text.
    - Always use fig.
    - Always use plotly if necessary or asked for a graph, else print only the value.
    - Always start counting from the last record inside the dataset and not from the actual day.
    - Always put the answer in a sentence.
        
    Context retrieved:
    {context}
        
    Question: {question}  based on the provided data in the prompt.
    Answer:
    """
    formatted_prompt = prompte_template.format(
        data=data,
        context="{context}",  # Leave as placeholder for later use
        question="{question}"  # Leave as placeholder for later use
    )
    return formatted_prompt.format(data=data, context="{context}", question="{question}")

# Function to extract python code from response
def extract_code(response):
    try:
        # Check if the response is a dictionary
        if isinstance(response, dict):
            # Ensure the dictionary has an 'answer' key
            if 'answer' in response:
                answer_text = response['answer']
            else:
                return "Error: Response dictionary does not contain 'answer' key."
        else:
            # Assume response is a string
            answer_text = response
        
        # Extract Python code using the appropriate delimiters
        start = answer_text.find("```python") + len("```python")
        end = answer_text.find("```", start)
        python_code = answer_text[start:end].strip()

        # Check if any code was extracted
        if python_code:
            return python_code
        else:
            return "Error: No Python code found in the response."
            
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

def handle_agent_query_rag(data, query):
    try:

        # Create Prompt Template Object
        prompt_template = def_rag_prompt_template(data)
        print(prompt_template)
        qa_prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)

        # Initialize Memory
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        
        qa = ConversationalRetrievalChain.from_llm(
            llm=gllm,
            retriever=vectorestore.as_retriever(),  # Pass the PineconeVectorStore directly
            memory=memory,
            combine_docs_chain_kwargs={"prompt": qa_prompt}
        )
        
        response = qa({"question": query})
        
        if not response:
            return "Error: No response from the model."
        
        return response
    except Exception as e:
        return f"Error: {str(e)}"
    

@app.route('/send_predict', methods=['POST'])
def send_predict():
    data = request.json.get('data')
    query = request.json.get('query')
    response = handle_agent_query(data, query)
    return jsonify({'prediction': response})

@app.route('/rag_predict', methods=['POST'])
def rag_predict():
    data = request.json.get('data')
    query = request.json.get('query')

    if data is None or query is None:
        return jsonify({'error': 'Missing data or query'}), 400
    
    response = handle_agent_query_rag(data, query)
    response = extract_code(response)
    return jsonify({'prediction': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
