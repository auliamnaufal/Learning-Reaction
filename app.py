import os
import io
import pickle
import datetime
import pandas as pd
from flask import Flask, request, render_template, jsonify, send_file, abort
from flask_cors import CORS, cross_origin
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Load training data
TRAINING_DATA_PATH = 'Training Kategori - Sheet1.csv'
MODEL_PATH = "./predict_category_model.pkl"
STORAGE_DIR = "storage"

data_training = pd.read_csv(TRAINING_DATA_PATH)
texts = data_training['feedback'].tolist()

# Load trained category prediction model
category_model = pickle.load(open(MODEL_PATH, "rb"))

# Initialize vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
vectorizer.fit(texts)

# Load Sentiment Analysis Model
PRETRAINED_MODEL = 'mdhugol/indonesia-bert-sentiment-classification'
sentiment_model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL)
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
sentiment_pipeline = pipeline('sentiment-analysis', model=sentiment_model, tokenizer=tokenizer)

# Load Summarization Model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Category mapping
CATEGORY_MAP = {
    1: "Materi",
    2: "Jam mata kuliah",
    3: "Tugas",
    4: "Pembelajaran",
    5: "Ujian",
    6: "Lainnya"
}

LABEL_MAP = {
    'LABEL_0': 'Positif',
    'LABEL_1': 'Netral',
    'LABEL_2': 'Negatif'
}

# Ensure storage directory exists
os.makedirs(STORAGE_DIR, exist_ok=True)

# Helper Functions
def predict_sentiment(text):
    """Predicts sentiment label for a given text."""
    result = sentiment_pipeline(text)
    return LABEL_MAP[result[0]['label']]

def summarize_text(text):
    """Summarizes long text using a transformer model."""
    words = text.split()
    max_length = min(150, len(words) // 2)
    max_length = max(max_length, 30)

    summary = summarizer(text[:1024], max_length=max_length, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def process_file(file):
    """Reads the uploaded file and returns a pandas DataFrame."""
    filename = file.filename
    if filename.endswith('.csv'):
        return pd.read_csv(file)
    elif filename.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file)
    else:
        return None

# Routes
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/mini_overview")
def mini_overview():
    return render_template("mini_overview.html")

@app.route("/user_input")
def user_input():
    return render_template("user_input.html")

@app.route("/output")
def output():
    """Displays the summarized output file in a table."""
    filename = request.args.get("filename")
    if not filename:
        return abort(400, description="Filename parameter is missing.")

    file_path = os.path.join(STORAGE_DIR, filename)
    if not os.path.isfile(file_path):
        return abort(404, description="File not found.")

    try:
        df = pd.read_csv(file_path)
        data_summary = df.to_dict(orient="records")
        return render_template("output.html", data_summary=data_summary)
    except Exception as e:
        return abort(500, description=f"Error reading CSV file: {str(e)}")

@app.route("/download", methods=["POST"])
def download():
    """Allows users to download the summarized output file."""
    filename = request.args.get("filename")
    if not filename:
        return abort(400, description="Filename parameter is missing.")

    file_path = os.path.join(STORAGE_DIR, filename)
    if os.path.isfile(file_path):
        return send_file(file_path, as_attachment=True)

    return abort(404, description="File not found.")

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    """Processes uploaded reviews, predicts category & sentiment, and returns summarized results."""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        df = process_file(file)
        if df is None:
            return jsonify({"error": "Unsupported file format. Upload a CSV or Excel file."}), 400

        if 'review' not in df.columns:
            return jsonify({"error": "'review' column not found in file"}), 400

        # Process each review
        predictions = []
        for _, row in df.iterrows():
            review_text = str(row['review'])
            sentences = [s.strip() for s in review_text.split('.') if s.strip()]
            
            for sentence in sentences:
                text_vector = vectorizer.transform([sentence])
                category_prediction = category_model.predict(text_vector)[0]
                sentiment_prediction = predict_sentiment(sentence)

                predictions.append({
                    "predicted_category": CATEGORY_MAP.get(category_prediction, "Unknown"),
                    "label": sentiment_prediction,
                    "feedback": sentence
                })

        # Convert to DataFrame
        output_df = pd.DataFrame(predictions)

        # Group and summarize
        grouped_data = output_df.groupby(['predicted_category', 'label'])['feedback'].apply(' '.join).reset_index()
        grouped_data['summary'] = grouped_data['feedback'].apply(summarize_text)

        # Save the summarized output
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        output_filename = f"{file.filename.rsplit('.', 1)[0]}-output-{timestamp}.csv"
        output_filepath = os.path.join(STORAGE_DIR, output_filename)

        grouped_data[['predicted_category', 'label', 'summary']].to_csv(output_filepath, index=False)

        return jsonify({"message": "Predictions saved successfully", "file": output_filename})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
