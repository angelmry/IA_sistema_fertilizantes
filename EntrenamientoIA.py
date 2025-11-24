# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json

# -----------------------------
# 1. Cargar el dataset
# -----------------------------
df = pd.read_csv("Crop_and_fertilizer_dataset.csv")

# -----------------------------
# 2. Selección de variables
# -----------------------------
X = df.drop(['Fertilizer', 'Link'], axis=1)
y = df['Fertilizer']

# -----------------------------
# 3. Preprocesamiento
# -----------------------------
categorical_cols = ['District_Name', 'Soil_color', 'Crop']
le_dict = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le   # <-- guardar LabelEncoder

# -----------------------------
# 4. División
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 5. Entrenamiento (Balanceado y Mejorado)
# -----------------------------
clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=4,
    class_weight='balanced',
    random_state=42
)

clf.fit(X_train, y_train)

# -----------------------------
# 6. Evaluación
# -----------------------------
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

print("Accuracy:", accuracy)

# -----------------------------
# 7. Guardar modelo y label encoders
# -----------------------------
joblib.dump(clf, "fertilizer_prediction_model.pkl")
joblib.dump(le_dict, "label_encoders.pkl")

print("\nGuardado modelo y encoders correctamente.")

# -----------------------------
# 8. Guardar métricas
# -----------------------------
resultados_json = {
    "accuracy": accuracy,
    "classification_report": report
}

with open("resultados.json", "w") as f:
    json.dump(resultados_json, f, indent=4)

print("\nResultados guardados.")
