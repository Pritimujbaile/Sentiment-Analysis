from flask import Flask, request, jsonify, send_file
from project import predict_sentiment

app = Flask(__name__)

# Serve HTML from same folder
@app.route("/")
def home():
    return send_file("project.html")

# Receive input from HTML
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    review = data["review"]

    sentiment = predict_sentiment(review)

    return jsonify({"sentiment": sentiment})

if __name__ == "__main__":
    app.run(debug=True)
