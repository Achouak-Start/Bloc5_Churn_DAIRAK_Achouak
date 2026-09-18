import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ============================================
# 1. CHARGEMENT DU JEU DE DONNÉES RFM + CHURN
# ============================================
rfm = pd.read_csv("data/processed/rfm_churn.csv")

# On sépare les variables explicatives (X) de la variable cible (y)
# Customer ID est exclu : ce n'est pas un comportement d'achat, juste un identifiant
X = rfm[["Recence", "Frequence", "Montant"]]
y = rfm["Churn"]

print("Variables candidates :", list(X.columns))
print(f"Nombre d'observations : {X.shape[0]}")

# ============================================
# 2. ENTRAÎNEMENT RAPIDE D'UN RANDOM FOREST
#    (uniquement pour mesurer l'importance des variables,
#     pas encore le modèle final)
# ============================================
modele_exploratoire = RandomForestClassifier(random_state=42)
modele_exploratoire.fit(X, y)

# ============================================
# 3. CALCUL DE L'IMPORTANCE DE CHAQUE VARIABLE
# ============================================
importances = pd.DataFrame({
    "Variable": X.columns,
    "Importance": modele_exploratoire.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\n=== Importance des variables (méthode incorporée - Random Forest) ===")
print(importances)

# ============================================
# 4. SAUVEGARDE DU RÉSULTAT
# ============================================
importances.to_csv("outputs/resultats/importance_variables.csv", index=False)
print("\nRésultat sauvegardé dans outputs/resultats/importance_variables.csv")