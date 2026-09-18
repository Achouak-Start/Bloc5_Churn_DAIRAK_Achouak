from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_accueil():
    """Vérifie que la route d'accueil répond correctement"""
    reponse = client.get("/")
    assert reponse.status_code == 200

def test_prediction():
    """Vérifie que la route /predict renvoie une prédiction cohérente"""
    donnees_client = {
        "Recence": 300,
        "Frequence": 1,
        "Montant": 150
    }
    reponse = client.post("/predict", json=donnees_client)
    
    assert reponse.status_code == 200
    
    resultat = reponse.json()
    assert "churn_predit" in resultat
    assert resultat["churn_predit"] in [0, 1]
    assert "probabilite_churn" in resultat