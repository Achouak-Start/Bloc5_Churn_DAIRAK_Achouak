import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score

# ============================================
# 1. CHARGEMENT DES DONNÉES D'ENTRAÎNEMENT D'ORIGINE
#    (le "profil habituel" du modèle)
# ============================================
rfm = pd.read_csv("data/processed/rfm_churn.csv")

moyennes_origine = rfm[["Recence", "Frequence", "Montant"]].mean()

print("=== Profil moyen des clients à l'entraînement (référence) ===")
print(moyennes_origine)

# ============================================
# 2. SIMULATION D'UN NOUVEAU LOT DE DONNÉES
#    (en pratique, ce serait un nouvel export de vente)
# ============================================
np.random.seed(42)

nouveau_lot = rfm.copy()
# On simule un changement de comportement : les clients achètent
# en moyenne un peu moins souvent et dépensent un peu moins
nouveau_lot["Frequence"] = nouveau_lot["Frequence"] * 0.7
nouveau_lot["Montant"] = np.random.permutation(nouveau_lot["Montant"].values)

moyennes_nouveau_lot = nouveau_lot[["Recence", "Frequence", "Montant"]].mean()

print("\n=== Profil moyen du nouveau lot de clients ===")
print(moyennes_nouveau_lot)

# ============================================
# 3. CALCUL DE L'ÉCART EN POURCENTAGE (détection de dérive)
# ============================================
ecart_pourcentage = ((moyennes_nouveau_lot - moyennes_origine) / moyennes_origine) * 100

print("\n=== Écart entre le nouveau lot et la référence (en %) ===")
print(ecart_pourcentage.round(2))

# Seuil d'alerte : si une variable dérive de plus de 15%
SEUIL_ALERTE = 15

print("\n=== Résultat de la surveillance ===")
alerte_declenchee = False
for variable, ecart in ecart_pourcentage.items():
    if abs(ecart) > SEUIL_ALERTE:
        print(f"⚠️ ALERTE : la variable '{variable}' a dérivé de {ecart:.1f}% (seuil : {SEUIL_ALERTE}%)")
        alerte_declenchee = True
    else:
        print(f"✅ '{variable}' reste stable ({ecart:.1f}%)")

if alerte_declenchee:
    print("\n🔴 Une dérive significative a été détectée : une réévaluation du modèle est recommandée.")
else:
    print("\n🟢 Aucune dérive significative détectée : le modèle reste fiable.")

# ============================================
# 4. VÉRIFICATION DE LA PERFORMANCE SUR LE NOUVEAU LOT
#    (ici on connaît le vrai Churn, donc on peut mesurer l'AUC)
# ============================================
modele = joblib.load("models/modele_churn_final.pkl")

X_nouveau = nouveau_lot[["Recence", "Frequence", "Montant"]]
y_nouveau = nouveau_lot["Churn"]

y_proba_nouveau = modele.predict_proba(X_nouveau)[:, 1]
auc_nouveau = roc_auc_score(y_nouveau, y_proba_nouveau)

print(f"\n=== Performance du modèle sur le nouveau lot ===")
print(f"AUC sur les données d'entraînement (référence) : 0.814")
print(f"AUC sur le nouveau lot                         : {auc_nouveau:.3f}")

# ============================================
# 5. SAUVEGARDE DU RAPPORT DE MONITORING
# ============================================
rapport = pd.DataFrame({
    "Variable": ecart_pourcentage.index,
    "Ecart_pourcentage": ecart_pourcentage.values
})
rapport.to_csv("outputs/resultats/rapport_monitoring.csv", index=False)
print("\nRapport de monitoring sauvegardé dans outputs/resultats/rapport_monitoring.csv")