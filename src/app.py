from flask import Flask, request, jsonify
from src.inference import predict

app = Flask(__name__)

@app.route('/')
def home():
    return "Raga Classification API Running!"

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.json['features']
    result = predict(data, 'models/model.pth')
    return jsonify({'prediction': int(result)})

if __name__ == "__main__":
    app.run(debug=True)