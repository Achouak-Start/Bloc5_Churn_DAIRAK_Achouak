from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# ============================================
# 1. CHARGEMENT DU MODÈLE SAUVEGARDÉ (lien avec C5.3.1)
# ============================================
modele = joblib.load("models/modele_churn_final.pkl")

# ============================================
# 2. CRÉATION DE L'API
# ============================================
app = FastAPI(title="API de prédiction du churn client")

# ============================================
# 3. DÉFINITION DE LA FORME DES DONNÉES ATTENDUES
#    (ce que le "guichet" doit recevoir pour fonctionner)
# ============================================
class ClientData(BaseModel):
    Recence: int
    Frequence: int
    Montant: float

# ============================================
# 4. ROUTE DE VÉRIFICATION (juste pour tester que l'API est en vie)
# ============================================
@app.get("/")
def accueil():
    return {"message": "API de prédiction du churn - en ligne"}

# ============================================
# 5. ROUTE DE PRÉDICTION (le vrai "guichet")
# ============================================
@app.post("/predict")
def predire_churn(client: ClientData):
    # On transforme les données reçues en tableau exploitable par le modèle
    donnees = pd.DataFrame([{
        "Recence": client.Recence,
        "Frequence": client.Frequence,
        "Montant": client.Montant
    }])

    prediction = modele.predict(donnees)[0]
    probabilite = modele.predict_proba(donnees)[0][1]

    return {
        "churn_predit": int(prediction),
        "probabilite_churn": round(float(probabilite), 3),
        "interprétation": "Client à risque" if prediction == 1 else "Client fidèle"
    }