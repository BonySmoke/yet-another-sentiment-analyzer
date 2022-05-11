from flask import request, jsonify, render_template
from sentiment import app
from sentiment.processor import Process

API_PATH = "/api/"
VERSION = "v1"


def parse_sentiment_request(data):
    if not data['text']:
        return jsonify({"error": "no text supplied"})
    process = Process()
    if data['clf_type']:
        process.clf_type = data['clf_type']
    if data['lang']:
        process.lang = data['lang']
    if data['algorithm']:
        process.algorithm = data['algorithm']
    return process


@app.route('/', methods=["GET", "POST"])
def home():
    default_data = {
        "text": "",
        "languages": [
            {"title": "English", "value": "en"},
            {"title": "Ukrainian", "value": "uk"}
        ],
        "algorithms": [
            {"title": "Naive Bayes", "value": "nb"},
            {"title": "Random Forest", "value": "forest"},
            {"title": "Linear Regression", "value": "linear"}
        ],
        "clf_types": [
            {"title": "Binary", "value": "binary"},
            {"title": "Three-classes", "value": "three"},
            {"title": "Fine grained", "value": "fine"}
        ],
        "explain": False
    }

    if request.method == "POST":
        data = request.form
        process = parse_sentiment_request(data)
        process._prepare_model()
        result = process.make_prediction(data['text'])
        default_data.update({
            "text": data['text'],
            "sentiment": result
        })

        if request.form.get('explain'):
            process.explain()
            default_data.update({"explain": True})
        print(default_data)
        return render_template('sentiment.html', data=default_data)

    return render_template('sentiment.html', data=default_data)


@app.route(f'{API_PATH}{VERSION}/sentiment', methods=["GET", "POST"])
def sentiment():
    data = request.get_json()
    process = parse_sentiment_request(data)

    process._prepare_model()

    result = process.make_prediction(data['text'])
    return jsonify({"result": result})
