import pandas as pd
import joblib
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# ============================================
# 1. CHARGEMENT ET ENTRAÎNEMENT
# ============================================
rfm = pd.read_csv("data/processed/rfm_churn.csv")
X = rfm[["Recence", "Frequence", "Montant"]]
y = rfm["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

modele = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42)
modele.fit(X_train, y_train)

# ============================================
# 2. ÉVALUATION DE LA PERFORMANCE
# ============================================
y_proba = modele.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_proba)

print(f"AUC obtenu : {auc:.3f}")

SEUIL_MINIMUM = 0.75

if auc < SEUIL_MINIMUM:
    print(f"❌ Performance insuffisante (AUC {auc:.3f} < seuil {SEUIL_MINIMUM})")
    sys.exit(1)
else:
    print(f"✅ Performance validée (AUC {auc:.3f} >= seuil {SEUIL_MINIMUM})")

# ============================================
# 3. SAUVEGARDE DU MODÈLE (pour l'étape "registre")
# ============================================
joblib.dump(modele, "models/modele_churn_final.pkl")
print("Modèle sauvegardé et prêt pour l'enregistrement dans le registre")
