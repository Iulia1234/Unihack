import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import joblib
import os

# ==========================
# 🟦 1. ÎNCĂRCARE DATE
# ==========================

def load_uv_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["target_log_reduction"])
    y = df["target_log_reduction"]
    return X, y

def load_potability_data(path):
    df = pd.read_csv(path)

    # === Aproximare Solids & Turbidity ===
    solids_mean = df["Solids"].mean()
    turb_mean = df["Turbidity"].mean()

    df["Solids"] = solids_mean
    df["Turbidity"] = turb_mean

    # Scoatem complet parametrii din model
    X = df.drop(columns=["Potability", "Solids", "Turbidity"])
    y = df["Potability"]
    return X, y



# ==========================
# 🟦 2. TRAIN MODELE UV
# ==========================

def train_uv_models():
    print("Antrenez modelele UV...")

    X, y = load_uv_data("fisier1_spectre.txt")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "uv_scaler.pkl")

    # ---- REGRESIE ----
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_scaled, y, test_size=0.2
    )

    reg = RandomForestRegressor()
    reg.fit(X_train_reg, y_train_reg)
    joblib.dump(reg, "model_uv_regression.pkl")

    # ---- CLASIFICARE ----
    threshold = 4
    y_class = (y >= threshold).astype(int)

    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
        X_scaled, y_class, test_size=0.2
    )

    cls = RandomForestClassifier()
    cls.fit(X_train_cls, y_train_cls)
    joblib.dump(cls, "model_uv_classification.pkl")

    print("✔ Modele UV antrenate.\n")

# ==========================
# 🟦 3. TRAIN MODELU POTABILITATE
# ==========================

def train_potability_model():
    print("Antrenez modelul de potabilitate...")

    X, y = load_potability_data("fisier2_spectre.txt")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, "potability_scaler.pkl")

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    joblib.dump(model, "model_potability.pkl")

    print("✔ Modelul de potabilitate antrenat.\n")


# ==========================
# 🟦 4. FUNCTII VERDICT
# ==========================

def verdict_uv(value, class_result):
    if class_result == 1:
        return f"Filtru UV OK ✓ — eficiență estimată: {value:.2f} log"
    else:
        return f"Filtru UV PROBLEMĂ ⚠ — eficiență estimată: {value:.2f} log"

def verdict_pot(pred):
    return "Apa este POTABILĂ ✓" if pred == 1 else "Apa NU este potabilă ⚠"


# ==========================
# 🟦 5. PREDICT
# ==========================

def predict_all():
    print("Realizez predicția finală...\n")

    # Încărcăm modelele
    uv_scaler = joblib.load("uv_scaler.pkl")
    pot_scaler = joblib.load("potability_scaler.pkl")

    uv_reg = joblib.load("model_uv_regression.pkl")
    uv_cls = joblib.load("model_uv_classification.pkl")
    pot_model = joblib.load("model_potability.pkl")

    # Exemplu de input (pune tu aici date reale când le ai)
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

    water_input = pd.DataFrame([{
        "ph": 7.2,
        "Hardness": 190,
        "Solids": 15000,
        "Conductivity": 420,
        "Turbidity": 2
    }])

    # --- UV PRED ---
    uv_scaled = uv_scaler.transform(uv_input)
    pred_reg = uv_reg.predict(uv_scaled)[0]
    pred_cls = uv_cls.predict(uv_scaled)[0]

    # --- POTABILITY PRED ---
    water_scaled = pot_scaler.transform(water_input)
    pred_pot = pot_model.predict(water_scaled)[0]

    # --- RESULTATE ---
    print("=== VERDICT FINAL ===\n")
    print(verdict_uv(pred_reg, pred_cls))
    print(verdict_pot(pred_pot))


# ==========================
# 🟦 6. RUN
# ==========================

if __name__ == "__main__":

    # Dacă modelele nu există -> le antrenăm automat
    if not (os.path.exists("model_uv_regression.pkl") and
            os.path.exists("model_uv_classification.pkl")):
        train_uv_models()

    if not os.path.exists("model_potability.pkl"):
        train_potability_model()

    # După antrenare, putem prezice
    predict_all()
