from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Charger le modèle
model = joblib.load("model/pipeline_simple.pkl")

# Créer l'app FastAPI
app = FastAPI()

# Schéma de données attendu
class ClientData(BaseModel):
    revenu: float
    age: float
    anciennete: float
    nb_incidents: float

# Route de prédiction
@app.post("/predict")
def predict_score(data: ClientData):
    df = pd.DataFrame([{
        "revenu": data.revenu,
        "age": data.age,
        "anciennete": data.anciennete,
        "nb_incidents": data.nb_incidents
    }])
    
    proba = model.predict_proba(df)[0, 1]
    return {"score_credit": float(round(proba, 4))}

