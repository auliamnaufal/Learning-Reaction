from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline , AutoTokenizer, AutoModelForSequenceClassification
import os

app = Flask(__name__)

# Load training data for consistent vectorization
data_training = pd.read_csv('Training Kategori - Sheet1.csv')
texts = data_training['feedback'].tolist()

# Load trained model
category_model = pickle.load(open("./predict_category_model.pkl", "rb"))

# Initialize vectorizer with training data
vectorizer = TfidfVectorizer(max_features=5000)
vectorizer.fit(texts)

pretrained = 'mdhugol/indonesia-bert-sentiment-classification'

sentiment_model = AutoModelForSequenceClassification.from_pretrained(pretrained)
tokenizer = AutoTokenizer.from_pretrained(pretrained)

sentiment_analysis = pipeline('sentiment-analysis', model=sentiment_model, tokenizer=tokenizer)

def label_text(teks):
  results = sentiment_analysis(teks)
  label_text = {
      'LABEL_0' : 'positif',
      'LABEL_1' : 'netral',
      'LABEL_2' : 'negatif'
  }
  key = results[0]['label']
  return label_text[key]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        filename = file.filename

        if filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        else:
            return jsonify({"error": "Unsupported file format. Upload a CSV or Excel file."}), 400

        if 'review' not in df.columns:
            return jsonify({"error": "'review' column not found in file"}), 400

        output_data = []
        for _, row in df.iterrows():
            review_text = str(row['review'])
            sentences = [s.strip() for s in review_text.split('.') if s.strip()]
            
            category_predictions = []
            sentiment_predictions = []
            
            for sentence in sentences:
                text_transformed = vectorizer.transform([sentence])
                category_prediction = category_model.predict(text_transformed)[0]
                sentiment_prediction = label_text(sentence)
                
                category_predictions.append(category_prediction)
                sentiment_predictions.append(sentiment_prediction)

            for sentence, category, sentiment in zip(sentences, category_predictions, sentiment_predictions):
                output_data.append([sentence, category, sentiment])

        output_df = pd.DataFrame(output_data, columns=['Sentence', 'Category', 'Sentiment'])
        output_filename = "predictions.csv"
        output_df.to_csv(output_filename, index=False)

        return jsonify({"message": "Predictions saved successfully", "file": output_filename})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
