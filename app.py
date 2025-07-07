from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os

from summarizer import generate_summary, generate_title

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html', title=None, summary=None, error=None)


@app.route('/summarize', methods=['POST'])
def summarize_form():
    try:
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                return render_template('index.html', error="Unsupported file format")
        else:
            content = request.form['content']

        top_n = int(request.form.get('top_n', 5))
        summary_sentences = generate_summary(content, top_n)
        title = generate_title(summary_sentences)
        return render_template('index.html', title=title, summary=summary_sentences)
    except Exception as e:
        return render_template('index.html', error=f"An error occurred: {e}")


@app.route('/api/summarize', methods=['POST'])
def summarize_api():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" field in request'}), 400

        content = data['text']
        top_n = data.get('top_n', 5)
        summary_sentences = generate_summary(content, top_n)
        title = generate_title(summary_sentences)

        return jsonify({
            'title': title,
            'summary': summary_sentences
        })

    except Exception as e:
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)
