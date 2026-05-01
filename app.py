"""
Flask application for Crop Disease Prediction with interactive UI.
"""
import json
from flask import Flask, render_template, request, jsonify
from model_handler import ModelHandler

app = Flask(__name__)
handler = ModelHandler()


@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html', crops=list(handler.le_crop.classes_))


@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for disease prediction."""
    try:
        data = request.json
        df_input = handler.prepare_input(data)
        result = handler.predict(df_input)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/feature-importance', methods=['GET'])
def feature_importance():
    """API endpoint for global feature importance."""
    try:
        imp = handler.get_feature_importance()
        return jsonify(imp)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/local-contributions', methods=['POST'])
def local_contributions():
    """API endpoint for local perturbation-based contributions."""
    try:
        data = request.json
        df_input = handler.prepare_input(data)
        result = handler.get_local_contributions(df_input)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/sensitivity', methods=['POST'])
def sensitivity():
    """API endpoint for sensitivity analysis on a feature."""
    try:
        data = request.json
        df_input = handler.prepare_input(data)
        feature = data.get('feature')
        if not feature:
            return jsonify({'error': 'Feature name required'}), 400
        
        result = handler.get_sensitivity(df_input, feature)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)
