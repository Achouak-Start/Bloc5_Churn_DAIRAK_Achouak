import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

# ============================================
# 1. CHARGEMENT ET DÉCOUPAGE 
# ============================================
rfm = pd.read_csv("data/processed/rfm_churn.csv")
X = rfm[["Recence", "Frequence", "Montant"]]
y = rfm["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================
# 2. OPTIMISATION — RÉGRESSION LOGISTIQUE
# ============================================
print("="*50)
print("OPTIMISATION : RÉGRESSION LOGISTIQUE")
print("="*50)

grille_logreg = {
    "C": [0.01, 0.1, 1, 10, 100]
}

recherche_logreg = GridSearchCV(
    LogisticRegression(random_state=42),
    param_grid=grille_logreg,
    cv=5,
    scoring="roc_auc"
)
recherche_logreg.fit(X_train, y_train)

print(f"Meilleur paramètre C : {recherche_logreg.best_params_}")
print(f"Meilleur score AUC (validation croisée) : {recherche_logreg.best_score_:.3f}")

meilleur_logreg = recherche_logreg.best_estimator_
y_proba = meilleur_logreg.predict_proba(X_test)[:, 1]
print(f"AUC sur le jeu de test : {roc_auc_score(y_test, y_proba):.3f}")

# ============================================
# 3. OPTIMISATION — RANDOM FOREST
# ============================================
print("\n" + "="*50)
print("OPTIMISATION : RANDOM FOREST")
print("="*50)

grille_rf = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 10, None]
}

recherche_rf = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=grille_rf,
    cv=5,
    scoring="roc_auc"
)
recherche_rf.fit(X_train, y_train)

print(f"Meilleurs paramètres : {recherche_rf.best_params_}")
print(f"Meilleur score AUC (validation croisée) : {recherche_rf.best_score_:.3f}")

meilleur_rf = recherche_rf.best_estimator_
y_proba_rf = meilleur_rf.predict_proba(X_test)[:, 1]
print(f"AUC sur le jeu de test : {roc_auc_score(y_test, y_proba_rf):.3f}")

# ============================================
# 4. COMPARAISON AVANT / APRÈS OPTIMISATION
# ============================================
print("\n" + "="*50)
print("COMPARAISON AVANT / APRÈS OPTIMISATION")
print("="*50)

comparaison = pd.DataFrame({
    "Modele": ["Régression logistique (avant)", "Régression logistique (après)",
               "Random Forest (avant)", "Random Forest (après)"],
    "AUC": [0.811, roc_auc_score(y_test, y_proba),
            0.772, roc_auc_score(y_test, y_proba_rf)]
})
print(comparaison)

comparaison.to_csv("outputs/resultats/optimisation_avant_apres.csv", index=False)
print("\nComparaison sauvegardée dans outputs/resultats/optimisation_avant_apres.csv")