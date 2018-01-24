from flask import Flask, request, render_template
app = Flask(__name__)
from main import get_summary

def get_html(filename):
    return open(filename, 'r').read()


@app.route('/')
def main_page():
    return get_html('./static/index.html')


@app.route('/summary/')
def search():
    text = request.args.get('text', '')
    return render_template('summary.html', summary=get_summary(text))


if __name__ == '__main__':
    app.run(debug=True)
