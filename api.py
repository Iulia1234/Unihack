from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# =============================
#  Încarcare modele ML
# =============================
print("Incarc modelele...")

uv_scaler = joblib.load("uv_scaler.pkl")
pot_scaler = joblib.load("potability_scaler.pkl")

uv_reg = joblib.load("model_uv_regression.pkl")
uv_cls = joblib.load("model_uv_classification.pkl")
pot_model = joblib.load("model_potability.pkl")

print("Modele incarcate cu succes!")


# ==========================================================
#                ENDPOINT PRINCIPAL /predict
# ==========================================================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        print("JSON primit:", data)

        # -----------------------------
        # 1) Preluare valori din UI
        # -----------------------------
        ph = float(data.get("ph"))
        conductivity = float(data.get("conductivity"))
        hardness = float(data.get("hardness"))

        # -----------------------------
        # 2) Aproximare Solids + Turbidity
        #
        # Formulae simple (pot fi calibrate mai tarziu):
        # Solids ≈ conductivity * 2.5
        # Turbidity ≈ hardness / 150
        # -----------------------------
        solids = conductivity * 2.5
        turbidity = hardness / 150

        print(f"Valori generate → Solids={solids}, Turbidity={turbidity}")

        # -----------------------------
        # 3) Pregatire input pentru model
        # -----------------------------
        water_input = pd.DataFrame([{
            "ph": ph,
            "Solids": solids,
            "Turbidity": turbidity
        }])

        water_scaled = pot_scaler.transform(water_input)
        pred_pot = int(pot_model.predict(water_scaled)[0])
        confidence = float(pot_model.predict_proba(water_scaled)[0].max()) * 100

        # -----------------------------
        # 4) UV Simulat (dupa modelul tau)
        # -----------------------------
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
        uv_efficiency = float(uv_reg.predict(uv_scaled)[0])
        uv_status = int(uv_cls.predict(uv_scaled)[0])

        # -----------------------------
        # 5) Raspuns API
        # -----------------------------
        return jsonify({
            "isPotable": bool(pred_pot == 1),
            "confidence": round(confidence, 2),

            "input": {
                "ph": ph,
                "conductivity": conductivity,
                "hardness": hardness,
                "solids": solids,
                "turbidity": turbidity
            },

            "uv": {
                "status": "operational" if uv_status == 1 else "warning",
                "efficiency": round(uv_efficiency, 2)
            }
        })

    except Exception as e:
        print("EROARE API:", str(e))
        return jsonify({"error": str(e)}), 500


# ==========================================================
#                  HEALTH CHECK
# ==========================================================
@app.route('/health')
def health():
    return jsonify({"status": "ok"})


# ==========================================================
#                    RUN SERVER
# ==========================================================
if __name__ == '__main__':
    print("Server pornit → http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
