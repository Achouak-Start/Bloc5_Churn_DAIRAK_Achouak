import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# ============================================
# 1. CHARGEMENT ET DÉCOUPAGE (identique aux scripts précédents)
# ============================================
rfm = pd.read_csv("data/processed/rfm_churn.csv")
X = rfm[["Recence", "Frequence", "Montant"]]
y = rfm["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================
# 2. ENTRAÎNEMENT DU MODÈLE FINAL
#    (Random Forest avec les meilleurs hyperparamètres trouvés en C5.2.4)
# ============================================
print("Entraînement du modèle final (Random Forest optimisé)...")

modele_final = RandomForestClassifier(
    n_estimators=300,
    max_depth=5,
    random_state=42
)
modele_final.fit(X_train, y_train)

y_proba = modele_final.predict_proba(X_test)[:, 1]
auc_avant_sauvegarde = roc_auc_score(y_test, y_proba)
print(f"AUC du modèle final avant sauvegarde : {auc_avant_sauvegarde:.3f}")

# ============================================
# 3. SAUVEGARDE DU MODÈLE (sérialisation avec joblib)
# ============================================
chemin_modele = "models/modele_churn_final.pkl"
joblib.dump(modele_final, chemin_modele)
print(f"\nModèle sauvegardé dans : {chemin_modele}")

# ============================================
# 4. VÉRIFICATION : RECHARGER LE MODÈLE ET COMPARER
# ============================================
print("\nVérification : rechargement du modèle sauvegardé...")
modele_recharge = joblib.load(chemin_modele)

y_proba_recharge = modele_recharge.predict_proba(X_test)[:, 1]
auc_apres_rechargement = roc_auc_score(y_test, y_proba_recharge)
print(f"AUC du modèle rechargé : {auc_apres_rechargement:.3f}")

if auc_avant_sauvegarde == auc_apres_rechargement:
    print("\n✅ Le modèle rechargé donne exactement les mêmes résultats : sauvegarde réussie.")
else:
    print("\n⚠️ Différence détectée entre le modèle original et le modèle rechargé.")