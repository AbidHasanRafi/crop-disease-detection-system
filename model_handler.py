"""
Model prediction and analysis logic shared between Streamlit and Flask apps.
"""
import os
import joblib
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve


class ModelHandler:
    """Encapsulates model loading, prediction, and analysis."""
    
    def __init__(self, model_dir='model'):
        """Load all trained artifacts from model_dir (default: ./model directory)."""
        # If model_dir is relative, resolve it from the current file's location
        if not os.path.isabs(model_dir):
            model_dir = os.path.join(os.path.dirname(__file__), model_dir)
        
        self.model = joblib.load(os.path.join(model_dir, 'lgbm_tuned_model.pkl'))
        self.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        
        with open(os.path.join(model_dir, 'label_encoders.pkl'), 'rb') as f:
            encoders = pickle.load(f)
        self.le_crop = encoders['crop']
        self.le_disease = encoders['disease']
        
        with open(os.path.join(model_dir, 'feature_cols.txt'), 'r') as f:
            self.feature_cols = f.read().strip().split(',')
        
        self.optimal_thresholds = np.load(os.path.join(model_dir, 'optimal_thresholds.npy'))
    
    def prepare_input(self, params_dict):
        """
        Prepare input DataFrame from raw parameters.
        
        Args:
            params_dict: dict with keys like 'nitrogen', 'phosphorus', etc.
        
        Returns:
            df_input: DataFrame ready for model.predict_proba
        """
        # Extract and convert parameters
        crop = params_dict.get('crop', self.le_crop.classes_[0])
        nitrogen = float(params_dict.get('nitrogen', 33.71))
        phosphorus = float(params_dict.get('phosphorus', 42.91))
        potassium = float(params_dict.get('potassium', 71.63))
        ph_value = float(params_dict.get('ph_value', 5.64))
        soil_temp = float(params_dict.get('soil_temp', 12.61))
        humidity = float(params_dict.get('humidity', 68.97))
        rainfall = float(params_dict.get('rainfall', 183.16))
        weather_temp = float(params_dict.get('weather_temp', 82.70))
        weather_humidity = float(params_dict.get('weather_humidity', 89.90))
        cloud_cover = float(params_dict.get('cloud_cover', 98.60))
        wind_speed = float(params_dict.get('wind_speed', 9.60))
        rolling_rainfall = float(params_dict.get('rolling_rainfall', 89.89))
        temp_variance = float(params_dict.get('temp_variance', 9.60))
        favorability = float(params_dict.get('favorability', 0.1746))
        
        crop_encoded = self.le_crop.transform([crop])[0]
        
        input_dict = {
            'Nitrogen': nitrogen,
            'Phosphorus': phosphorus,
            'Potassium': potassium,
            'pH_Value': ph_value,
            'Soil_Temperature': soil_temp,
            'Humidity': humidity,
            'Rainfall': rainfall,
            'Weather_Temp_F': weather_temp,
            'Weather_Humidity': weather_humidity,
            'Cloud_Cover_pct': cloud_cover,
            'Wind_Speed_kmh': wind_speed,
            'Rolling_Rainfall_mm': rolling_rainfall,
            'Temp_Variance': temp_variance,
            'Weather_Favorability_Score': favorability,
            'Crop_Encoded': crop_encoded
        }
        
        # Compute engineered features
        input_dict['Thermal_Gap'] = soil_temp - weather_temp
        input_dict['Moisture_Gap'] = rainfall - humidity
        input_dict['Stress_Index'] = temp_variance * (1 - favorability)
        input_dict['Crop_Weather'] = crop_encoded * favorability
        input_dict['Thermal_Product'] = soil_temp * weather_temp
        input_dict['Moisture_Product'] = rainfall * humidity
        input_dict['Wind_Rain_Product'] = wind_speed * rolling_rainfall
        input_dict['Crop_Temp'] = crop_encoded * soil_temp
        
        # Fill missing features
        for col in self.feature_cols:
            if col not in input_dict:
                input_dict[col] = 0.0
        
        df_input = pd.DataFrame([input_dict])[self.feature_cols]
        return df_input
    
    def predict(self, df_input):
        """
        Predict disease class and probabilities.
        
        Returns:
            dict with 'disease', 'confidence', 'probabilities'
        """
        X_scaled_array = self.scaler.transform(df_input)
        X_scaled = pd.DataFrame(X_scaled_array, columns=self.feature_cols)
        proba = self.model.predict_proba(X_scaled)[0]
        
        pred_class = np.argmax(proba >= self.optimal_thresholds)
        disease = self.le_disease.inverse_transform([pred_class])[0]
        confidence = float(proba[pred_class])
        
        return {
            'disease': disease,
            'confidence': confidence,
            'probabilities': {str(cls): float(prob) for cls, prob in zip(self.le_disease.classes_, proba)},
            'top_3': self._get_top_3(proba)
        }
    
    def _get_top_3(self, proba):
        """Get top 3 disease predictions."""
        top3_idx = np.argsort(proba)[-3:][::-1]
        return [
            {'disease': self.le_disease.inverse_transform([idx])[0], 'probability': float(proba[idx])}
            for idx in top3_idx
        ]
    
    def get_feature_importance(self):
        """Get global feature importance (top 20)."""
        try:
            if hasattr(self.model, 'feature_importances_'):
                imp = self.model.feature_importances_
            else:
                imp = np.array(self.model.booster_.feature_importance(importance_type='gain'))
            
            feat_imp = pd.Series(imp, index=self.feature_cols).sort_values(ascending=False)[:20]
            return {feat: float(val) for feat, val in feat_imp.items()}
        except Exception as e:
            return {'error': str(e)}
    
    def get_local_contributions(self, df_input):
        """Compute perturbation-based local contributions for predicted class."""
        base = df_input.iloc[0].copy()
        
        base_scaled = pd.DataFrame(self.scaler.transform(pd.DataFrame([base])), columns=self.feature_cols)
        base_proba = self.model.predict_proba(base_scaled)[0]
        pred_idx = int(np.argmax(base_proba))
        
        contributions = {}
        for f in self.feature_cols:
            x_plus = base.copy()
            val = float(base[f])
            delta = 0.1 * (abs(val) if abs(val) > 1e-6 else 1.0)
            x_plus[f] = val + delta
            x_plus_scaled = pd.DataFrame(self.scaler.transform(pd.DataFrame([x_plus])), columns=self.feature_cols)
            p_plus = self.model.predict_proba(x_plus_scaled)[0][pred_idx]
            contributions[f] = float(p_plus - base_proba[pred_idx])
        
        contrib_series = pd.Series({f: abs(c) for f, c in contributions.items()}).sort_values(ascending=False)
        top_feats = contrib_series.index[:8].tolist()
        
        result = {
            'predicted_class': self.le_disease.classes_[pred_idx],
            'top_features': top_feats,
            'contributions': {f: contributions[f] for f in top_feats}
        }
        return result
    
    def get_sensitivity(self, df_input, feature_name, n_points=50):
        """Generate sensitivity data for a single feature."""
        base = df_input.iloc[0].copy()
        
        base_scaled = pd.DataFrame(self.scaler.transform(pd.DataFrame([base])), columns=self.feature_cols)
        base_proba = self.model.predict_proba(base_scaled)[0]
        pred_idx = int(np.argmax(base_proba))
        
        val = float(base[feature_name])
        if abs(val) < 1e-6:
            vals = np.linspace(0, 1, n_points)
        else:
            vals = np.linspace(val * 0.5, val * 1.5, n_points)
        
        probs = []
        for v in vals:
            x_row = base.copy()
            x_row[feature_name] = v
            x_scaled = pd.DataFrame(self.scaler.transform(pd.DataFrame([x_row])), columns=self.feature_cols)
            probs.append(float(self.model.predict_proba(x_scaled)[0][pred_idx]))
        
        return {'feature': feature_name, 'values': vals.tolist(), 'probabilities': probs}
