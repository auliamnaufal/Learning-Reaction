import io
from flask import Flask, abort, render_template, request, jsonify, send_file
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline , AutoTokenizer, AutoModelForSequenceClassification
import os
import datetime

app = Flask(__name__)
CORS(app) # allow CORS for all domains on all routes.
app.config['CORS_HEADERS'] = 'Content-Type'

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
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Group data by 'predicted_category' and 'label'
# grouped_data = data.groupby(['predicted_category', 'label'])['feedback'].apply(' '.join).reset_index()

def label_text(teks):
  results = sentiment_analysis(teks)
  label_text = {
      'LABEL_0' : 'positif',
      'LABEL_1' : 'netral',
      'LABEL_2' : 'negatif'
  }
  key = results[0]['label']
  return label_text[key]

@app.route("/")
def home():
    return render_template("home.html")

# Route untuk mini overview learning reaction
@app.route("/mini_overview")
def mini_overview():
    return render_template("mini_overview.html")

# Route untuk Halaman Input User
@app.route("/user_input")
def user_input():
    return render_template("user_input.html")

# Route untuk menampilkan tabel summary
@app.route("/output")
def output():
    filename = request.args.get("filename")  # Get filename from request
    
    if not filename:
        return abort(400, description="Filename parameter is missing.")

    file_path = os.path.join("storage", filename)

    if not os.path.isfile(file_path):
        return abort(404, description="File not found.")

    # Read CSV file into a Pandas DataFrame
    try:
        df = pd.read_csv(file_path)

        # Convert DataFrame to list of dictionaries
        data_summary = df.to_dict(orient="records")
    except Exception as e:
        return abort(500, description=f"Error reading CSV file: {str(e)}")
    
    print(data_summary)

    return render_template("output.html", data_summary=data_summary)

# Route untuk mengunduh data sebagai CSV
@app.route("/download", methods=["POST"])
def download():
    filename = request.args.get("filename")  # Get filename from request

    if not filename:
        return abort(400, description="Filename parameter is missing.")

    file_path = os.path.join("storage/", filename)  # Change "your_directory" as needed

    if os.path.isfile(file_path):
        return send_file(file_path, as_attachment=True)

    return abort(404, description="File not found.")


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        filename = file.filename

        # Read file into DataFrame
        if filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        else:
            return jsonify({"error": "Unsupported file format. Upload a CSV or Excel file."}), 400

        if 'review' not in df.columns:
            return jsonify({"error": "'review' column not found in file"}), 400

        # Mapping of category numbers to text
        category_map = {
            1: "materi",
            2: "jam mata kuliah",
            3: "tugas",
            4: "pembelajaran",
            5: "ujian",
            6: "lainnya"
        }

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

            # Store category, sentiment, and original feedback
            for sentence, category, sentiment in zip(sentences, category_predictions, sentiment_predictions):
                output_data.append([category, sentiment, sentence])

        # Convert output data into a DataFrame
        output_df = pd.DataFrame(output_data, columns=['predicted_category', 'label', 'feedback'])

        # Replace category numbers with text labels
        output_df['predicted_category'] = output_df['predicted_category'].map(category_map)

        # Group data by 'predicted_category' and 'label' before summarization
        grouped_data = output_df.groupby(['predicted_category', 'label'])['feedback'].apply(' '.join).reset_index()
        print(grouped_data)

        # Function to summarize text
        def summarize_text(text):
            input_length = len(text.split())  # Count words
            max_length = min(150, input_length // 2)  # Adjust dynamically (increased from 150 to 200)
            max_length = max(max_length, 30)  # Ensure a reasonable minimum length (increased from 30 to 50)

            summary = summarizer(text[:1024], max_length=max_length, min_length=30, do_sample=False)  # Adjusted min_length
            return summary[0]['summary_text']

        # Apply summarization to grouped data
        grouped_data['summary'] = grouped_data['feedback'].apply(summarize_text)

        # Save the summarized output
        output_filename = filename.replace(".", f"-output-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.")
        output_filepath = os.path.join("storage", output_filename)
        grouped_data[['predicted_category', 'label', 'summary']].to_csv(output_filepath, index=False)

        return jsonify({"message": "Predictions saved successfully", "file": output_filename})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)
