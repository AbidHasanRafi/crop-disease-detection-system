# Crop Disease Detection

A modern, responsive web application for predicting crop diseases using advanced machine learning (LightGBM). This Flask app provides an intuitive interface with interactive visualizations and explainability features.

## Features

- **Disease Prediction**: Real-time predictions with confidence scores
- **Interactive Charts**: Plotly-based visualizations for:
  - Class probability distributions
  - Global feature importance
  - Local perturbation-based contributions
  - Feature sensitivity analysis
- **Responsive Design**: Mobile-friendly UI with professional styling
- **Explainability**: Understand model decisions through local and global interpretations
- **API-First Architecture**: Clean separation of backend logic and frontend

## Project Structure

```
flask_app/
├── app.py                      # Flask application & API endpoints
├── model_handler.py            # Model prediction logic
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── templates/
│   └── index.html             # Main HTML template
└── static/
    ├── style.css              # Responsive styling
    └── script.js              # Frontend interactions & Plotly charts
```

## Setup & Installation

### Prerequisites
- Python 3.8+
- Pre-trained model files in `model/` directory:
  - `lgbm_tuned_model.pkl`
  - `scaler.pkl`
  - `label_encoders.pkl`
  - `feature_cols.txt`
  - `optimal_thresholds.npy`

### Installation Steps

1. Navigate to the Flask app directory:
```bash
cd flask_app
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Ensure model files are in the parent `st_app` directory

## Running the Application

```bash
python app.py
```

The app will start on `http://localhost:5000` by default.

## Usage

1. **Input Parameters**: Fill in the sidebar form with:
   - Crop type selection
   - Soil properties (Nitrogen, Phosphorus, Potassium, pH)
   - Temperature & humidity values
   - Weather & rainfall data
   - Derived metrics

2. **Make Prediction**: Click "Predict Disease" to:
   - Get the predicted disease class and confidence
   - View top 3 disease predictions
   - See probability distribution chart
   - Analyze feature importance

3. **Analyze Results**: Use the interactive charts to:
   - Understand global feature importance
   - See local contributions for the predicted class
   - Analyze feature sensitivity by toggling the sensitivity analysis

## API Endpoints

### POST `/api/predict`
Predict disease class for given input parameters.

**Request Body**:
```json
{
  "crop": "Corn",
  "nitrogen": 33.71,
  "phosphorus": 42.91,
  ...
}
```

**Response**:
```json
{
  "disease": "Leaf_Blight",
  "confidence": 0.87,
  "probabilities": {"Leaf_Blight": 0.87, ...},
  "top_3": [{"disease": "Leaf_Blight", "probability": 0.87}, ...]
}
```

### GET `/api/feature-importance`
Get global feature importance (top 20).

**Response**:
```json
{
  "Rainfall": 0.156,
  "Weather_Favorability_Score": 0.142,
  ...
}
```

### POST `/api/local-contributions`
Get local perturbation-based contributions for predicted class.

**Request Body**: Same as `/api/predict`

**Response**:
```json
{
  "predicted_class": "Leaf_Blight",
  "top_features": ["Rainfall", "Humidity", ...],
  "contributions": {"Rainfall": 0.05, "Humidity": -0.02, ...}
}
```

### POST `/api/sensitivity`
Get sensitivity data for a specific feature (feature value vs predicted probability).

**Request Body**:
```json
{
  "crop": "Corn",
  "nitrogen": 33.71,
  ...,
  "feature": "Rainfall"
}
```

**Response**:
```json
{
  "feature": "Rainfall",
  "values": [91.58, 102.73, ...],
  "probabilities": [0.65, 0.70, ...]
}
```

## Architecture

### Backend (`app.py`)
- Flask application with RESTful API endpoints
- Routes for predictions and analysis
- Serves HTML template and static assets

### Model Handler (`model_handler.py`)
- Model artifact loading and caching
- Input preparation and feature engineering
- Prediction logic with probability thresholds
- Explainability computations:
  - Global feature importance
  - Local perturbation-based contributions
  - Feature sensitivity analysis

### Frontend (`templates/index.html`)
- Responsive form with input validation
- Real-time result display
- Sticky sidebar for easy input access

### Styling (`static/style.css`)
- Modern gradient backgrounds
- Responsive grid layout
- Professional color scheme
- Smooth transitions and hover effects
- Mobile-first design

### Interactivity (`static/script.js`)
- Form data serialization
- Parallel API requests
- Plotly chart rendering
- Dynamic sensitivity chart generation
- Error handling and loading states