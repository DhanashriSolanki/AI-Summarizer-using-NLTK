from flask import Flask, render_template, request
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import networkx as nx
from nltk.stem import WordNetLemmatizer
import re
import os
from werkzeug.utils import secure_filename

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def read_article(content):
    lemmatizer = WordNetLemmatizer()
    article = content.split(". ")
    sentences = []

    for sentence in article:
        cleaned_sentence = re.sub(r'[^a-zA-Z\s]', '', sentence)
        lemmatized_sentence = [lemmatizer.lemmatize(
            word.lower()) for word in cleaned_sentence.split()]
        sentences.append(lemmatized_sentence)

    if sentences and sentences[-1] == ['']:
        sentences.pop()

    return sentences


def sentence_similarity_tfidf(sent1, sent2):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(
        [' '.join(sent1), ' '.join(sent2)])
    return (tfidf_matrix * tfidf_matrix.T).toarray()[0, 1]


def gen_sim_matrix_tfidf(sentences):
    S = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                S[i][j] = sentence_similarity_tfidf(sentences[i], sentences[j])
    return S


def generate_summary(content, top_n=5):
    try:
        sentences = read_article(content)
        S = gen_sim_matrix_tfidf(sentences)

        sentence_similarity_graph = nx.from_numpy_array(S)
        scores = nx.pagerank(sentence_similarity_graph)

        ranked_sentence = sorted(
            ((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

        summarize_text = [
            " ".join(ranked_sentence[i][1]).capitalize() + "."
            for i in range(top_n)
        ]

        return summarize_text

    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def generate_title(summary_sentences):
    summary_text = ' '.join(summary_sentences)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5)
    X = vectorizer.fit_transform([summary_text])
    keywords = vectorizer.get_feature_names_out()
    return ' '.join(keywords).title()


@app.route('/')
def index():
    return render_template('index.html', title=None, summary=None, error=None)


@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                # Process the file content (assuming it's a text file)
                with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'r') as f:
                    content = f.read()
            else:
                return render_template('index.html', error="File format not allowed")
        else:
            content = request.form['content']

        top_n = int(request.form['top_n'])
        summary_sentences = generate_summary(content, top_n)
        title = generate_title(summary_sentences)
        return render_template('index.html', title=title, summary=summary_sentences)
    except Exception as e:
        return render_template('index.html', error=f"An error occurred: {e}")


if __name__ == '__main__':
    app.run(debug=True)
