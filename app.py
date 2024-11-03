from flask import Flask, render_template, request, redirect, url_for
import pickle
import re
import pandas as pd
from nltk.corpus import stopwords
import nltk

app = Flask(__name__)

# Load the pre-trained model and vectorizer
with open('./models/sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('./models/vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for processing user input and showing results
@app.route('/result', methods=['POST'])
def result():
    user_input = request.form['user_input']
    cleaned_text = clean_text(user_input)
    vectorized_text = vectorizer.transform([cleaned_text])
    sentiment = model.predict(vectorized_text)[0]
    sentiment_label = "Positive" if sentiment == 1 else "Negative"

    # Provide suggestions based on sentiment
    if sentiment == 1:
        suggestions = [
            "Celebrate your positive feelings! Plan a small gathering with friends.",
            "Reflect on things that make you happy.",
            "Consider journaling about what brings you joy.",
            "Keep doing activities that uplift you!"
        ]
    else:
        suggestions = [
            "I'm here for you. It might help to talk to someone.",
            "Consider taking a walk or a few deep breaths.",
            "Writing down your thoughts may provide relief.",
            "Don't hesitate to reach out for support."
        ]

    return render_template('result.html', sentiment=sentiment_label, suggestions=suggestions, user_input=user_input)

@app.route('/feedback', methods=['POST'])
def feedback():
    user_input = request.form['user_input']
    sentiment_label = request.form['sentiment_label']
    feedback_correct = request.form['correct']
    
    # If the user confirms the sentiment is correct
    if feedback_correct == 'yes':
        correct_sentiment = 1 if sentiment_label.lower() == "positive" else -1
        log_feedback(user_input, correct_sentiment)
    
    # If the user says the sentiment is incorrect, request correction
    elif feedback_correct == 'no':
        user_sentiment = request.form.get('user_sentiment')
        correct_sentiment = 1 if user_sentiment == 'positive' else -1
        log_feedback(user_input, correct_sentiment)
    
    # Redirect to the thanks page after feedback submission
    return redirect(url_for('thanks'))

@app.route('/thanks')
def thanks():
    return render_template('thanks.html')

def log_feedback(user_input, sentiment):
    feedback_df = pd.DataFrame({'review': [user_input], 'sentiment': [sentiment]})
    feedback_df.to_csv('feedback_log.csv', mode='a', header=False, index=False)

if __name__ == '__main__':
    app.run(port=5002, debug=True)
