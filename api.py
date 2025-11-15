from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# Incarca modelele antrenate
print("Incarc modelele...")
uv_scaler = joblib.load("uv_scaler.pkl")
pot_scaler = joblib.load("potability_scaler.pkl")
uv_reg = joblib.load("model_uv_regression.pkl")
uv_cls = joblib.load("model_uv_classification.pkl")
pot_model = joblib.load("model_potability.pkl")
print("Modele incarcate cu succes!")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print(f"Am primit date: {data}")
        
        # Extrage parametrii
        ph = float(data.get('ph', 7.0))
        conductivity = float(data.get('conductivity', 400))
        hardness = float(data.get('hardness', 150))
        
        # Creeaza DataFrame pentru potabilitate
        water_input = pd.DataFrame([{
            "ph": ph,
            "Hardness": hardness,
            "Conductivity": conductivity
        }])
        
        # Scaleaza si prezice potabilitatea
        water_scaled = pot_scaler.transform(water_input)
        pred_pot = pot_model.predict(water_scaled)[0]
        pred_proba = pot_model.predict_proba(water_scaled)[0]
        confidence = float(pred_proba.max() * 100)
        
        # Aproximare Solids & Turbidity
        solids = 15000.0
        turbidity = 3.5
        
        # Date UV simulate
        uv_input = pd.DataFrame([{
            "uv_power_mw_cm2": 1.5,
            "exposure_seconds": 300,
            "dose_mJ_cm2": 450,
            "distance_cm": 10,
            "temp_c": 20,
            "initial_cfu_log": 5,
            "D90_mJ_cm2": 4,
            "spectral_410nm": 900,
            "spectral_435nm": 920,
            "spectral_500nm": 850,
            "spectral_560nm": 880,
            "spectral_585nm": 820,
            "spectral_630nm": 760
        }])
        
        uv_scaled = uv_scaler.transform(uv_input)
        uv_efficiency = uv_reg.predict(uv_scaled)[0]
        uv_status = uv_cls.predict(uv_scaled)[0]
        
        # Construieste raspunsul
        response = {
            "isPotable": bool(pred_pot == 1),
            "confidence": round(confidence, 2),
            "approximated": {
                "solids": round(solids, 2),
                "turbidity": round(turbidity, 2)
            },
            "uv": {
                "status": "operational" if uv_status == 1 else "warning",
                "efficiency": round(float(uv_efficiency), 2)
            },
            "parameters": {
                "ph": ph,
                "conductivity": conductivity,
                "hardness": hardness
            }
        }
        
        print(f"Raspuns: {response}")
        return jsonify(response)
        
    except Exception as e:
        print(f"EROARE: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "message": "ML API is running"})

if __name__ == '__main__':
    print("\nStarting ML API Server...")
    print("Endpoint: http://localhost:5000/predict")
    print("Health check: http://localhost:5000/health\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
