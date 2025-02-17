from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load pre-trained model from Hugging Face
model = pipeline("sentiment-analysis")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data.get("text", "")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        # Perform inference
        result = model(text)

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
