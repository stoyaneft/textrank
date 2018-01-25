from flask import Flask, request, render_template
app = Flask(__name__)

from keywords import extract_keywords
from summarizer import summarize


def get_html(filename):
    return open(filename, 'r').read()


@app.route('/')
def main_page():
    return get_html('./static/index.html')


# @app.route('/summary/')
# def get_summary():
#     text = request.args.get('text', '')
#     sentences = request.args.get('sentences_count', 10)
#     return render_template('summary.html', summary=summarize(text, int(sentences)))


@app.route('/results/')
def get_results():
    text = request.args.get('text', '')
    sentences = request.args.get('sentences_count', 10)
    is_summary = request.args.get('summary')
    print(request.args)
    print('is', is_summary)
    if is_summary is not None:
        return render_template('summary.html', summary=summarize(text, int(sentences)))
    return render_template('keywords.html', keywords=extract_keywords(text, int(sentences)))


if __name__ == '__main__':
    app.run(debug=True)
