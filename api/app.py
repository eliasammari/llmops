from flask import Flask, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = Flask(__name__)

# Load the model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt")

    # Perform inference
    outputs = model(**inputs)
    prediction = outputs.logits.argmax().item()

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
