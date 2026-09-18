import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ============================================
# 1. CHARGEMENT DU JEU DE DONNÉES
# ============================================
rfm = pd.read_csv("data/processed/rfm_churn.csv")

X = rfm[["Recence", "Frequence", "Montant"]]
y = rfm["Churn"]

# ============================================
# 2. SÉPARATION ENTRAÎNEMENT / TEST (80% / 20%)
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Taille entraînement : {X_train.shape[0]} clients")
print(f"Taille test         : {X_test.shape[0]} clients")

# ============================================
# 3. MODÈLE 1 : RÉGRESSION LOGISTIQUE
# ============================================
print("\n" + "="*50)
print("MODÈLE 1 : RÉGRESSION LOGISTIQUE")
print("="*50)

modele_logreg = LogisticRegression(random_state=42)
modele_logreg.fit(X_train, y_train)

y_pred_logreg = modele_logreg.predict(X_test)
y_proba_logreg = modele_logreg.predict_proba(X_test)[:, 1]

print("\nRapport de classification :")
print(classification_report(y_test, y_pred_logreg))
print("Matrice de confusion :")
print(confusion_matrix(y_test, y_pred_logreg))
print(f"AUC (courbe ROC) : {roc_auc_score(y_test, y_proba_logreg):.3f}")

# ============================================
# 4. MODÈLE 2 : RANDOM FOREST
# ============================================
print("\n" + "="*50)
print("MODÈLE 2 : RANDOM FOREST")
print("="*50)

modele_rf = RandomForestClassifier(random_state=42)
modele_rf.fit(X_train, y_train)

y_pred_rf = modele_rf.predict(X_test)
y_proba_rf = modele_rf.predict_proba(X_test)[:, 1]

print("\nRapport de classification :")
print(classification_report(y_test, y_pred_rf))
print("Matrice de confusion :")
print(confusion_matrix(y_test, y_pred_rf))
print(f"AUC (courbe ROC) : {roc_auc_score(y_test, y_proba_rf):.3f}")

# ============================================
# 5. SAUVEGARDE DES RÉSULTATS
# ============================================
resultats = pd.DataFrame({
    "Modele": ["Régression logistique", "Random Forest"],
    "AUC": [
        roc_auc_score(y_test, y_proba_logreg),
        roc_auc_score(y_test, y_proba_rf)
    ]
})
resultats.to_csv("outputs/resultats/comparaison_modeles.csv", index=False)
print("\nComparaison des modèles sauvegardée dans outputs/resultats/comparaison_modeles.csv")