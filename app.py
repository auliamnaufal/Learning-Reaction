from flask import Flask, request, jsonify
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

app = Flask(__name__)


data_training = pd.read_csv('Training Kategori - Sheet1.csv')
texts = data_training['feedback'].tolist() 

category_model = pickle.load(open("./predict_category_model.pkl", "rb"))


# Initialize vectorizer with the fixed vocabulary
vectorizer = TfidfVectorizer(max_features=5000)
vectorizer.fit_transform(texts)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get text from request
        data = request.json
        texts = data.get("texts", [])

        if not texts or not isinstance(texts, list):
            return jsonify({"error": "Please provide a list of texts"}), 400

        # Transform text using the trained vectorizer
        text_transformed = vectorizer.transform(texts)
        print(text_transformed)

        # Predict category
        predictions = category_model.predict(text_transformed)

        return jsonify({"predictions": predictions.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
