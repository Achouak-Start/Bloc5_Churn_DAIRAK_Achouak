import subprocess
import sys
import pandas as pd
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# ============================================
# 1. ORCHESTRATION : ENCHAÎNEMENT AUTOMATIQUE DES ÉTAPES
#    (nettoyage -> construction des variables)
# ============================================
print("="*60)
print("DÉMARRAGE DU PIPELINE AUTOMATISÉ")
print("="*60)

etapes = [
    "scripts/01_nettoyage_donnees.py",
    "scripts/02_construction_variables.py"
]

for etape in etapes:
    print(f"\n>>> Exécution de {etape} ...")
    resultat = subprocess.run([sys.executable, etape], capture_output=True, text=True)
    
    if resultat.returncode != 0:
        print(f"❌ Erreur lors de l'exécution de {etape}")
        print(resultat.stderr)
        sys.exit(1)
    else:
        print(f"✅ {etape} terminé avec succès")

# ============================================
# 2. RÉ-ENTRAÎNEMENT AUTOMATIQUE DU MODÈLE FINAL
#    (avec les meilleurs hyperparamètres déjà trouvés en C5.2.4,
#     pas besoin de refaire tout le GridSearch à chaque fois)
# ============================================
print("\n>>> Ré-entraînement du modèle final...")

rfm = pd.read_csv("data/processed/rfm_churn.csv")
X = rfm[["Recence", "Frequence", "Montant"]]
y = rfm["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

modele = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42)
modele.fit(X_train, y_train)

y_proba = modele.predict_proba(X_test)[:, 1]
auc_actuel = roc_auc_score(y_test, y_proba)

print(f"✅ Modèle ré-entraîné - AUC obtenu : {auc_actuel:.3f}")

# ============================================
# 3. SAUVEGARDE DU NOUVEAU MODÈLE
# ============================================
joblib.dump(modele, "models/modele_churn_final.pkl")
print("✅ Modèle sauvegardé dans models/modele_churn_final.pkl")

# ============================================
# 4. HISTORISATION (le "carnet de bord" du pipeline)
# ============================================
chemin_historique = "outputs/resultats/historique_pipeline.csv"

nouvelle_ligne = pd.DataFrame({
    "Date_execution": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    "Nb_clients": [X.shape[0]],
    "AUC_obtenu": [round(auc_actuel, 3)]
})

try:
    historique = pd.read_csv(chemin_historique)
    historique = pd.concat([historique, nouvelle_ligne], ignore_index=True)
except FileNotFoundError:
    historique = nouvelle_ligne

historique.to_csv(chemin_historique, index=False)
print(f"\n✅ Exécution enregistrée dans l'historique : {chemin_historique}")

print("\n" + "="*60)
print("PIPELINE AUTOMATISÉ TERMINÉ AVEC SUCCÈS")
print("="*60)