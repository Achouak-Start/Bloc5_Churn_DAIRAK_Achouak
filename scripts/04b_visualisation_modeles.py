import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

# ============================================
# 1. CHARGEMENT ET RÉ-ENTRAÎNEMENT
#    (même split que le script 04, pour cohérence)
# ============================================
rfm = pd.read_csv("data/processed/rfm_churn.csv")
X = rfm[["Recence", "Frequence", "Montant"]]
y = rfm["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

modele_logreg = LogisticRegression(random_state=42)
modele_logreg.fit(X_train, y_train)

modele_rf = RandomForestClassifier(random_state=42)
modele_rf.fit(X_train, y_train)

# ============================================
# 2. MATRICES DE CONFUSION (les 2 modèles côte à côte)
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Palette adaptée daltonisme (viridis, comme au Bloc 2)
for ax, modele, nom in zip(
    axes,
    [modele_logreg, modele_rf],
    ["Régression logistique", "Random Forest"]
):
    y_pred = modele.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="viridis", ax=ax,
        xticklabels=["Fidèle", "Churné"],
        yticklabels=["Fidèle", "Churné"],
        cbar=False
    )
    ax.set_title(f"Matrice de confusion — {nom}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Prédiction du modèle")
    ax.set_ylabel("Valeur réelle")

plt.tight_layout()
plt.savefig("outputs/graphiques/matrices_confusion.png", dpi=150)
print("Matrices de confusion sauvegardées dans outputs/graphiques/matrices_confusion.png")
plt.close()

# ============================================
# 3. COURBES ROC (les 2 modèles superposées)
# ============================================
plt.figure(figsize=(7, 6))

for modele, nom, couleur in zip(
    [modele_logreg, modele_rf],
    ["Régression logistique", "Random Forest"],
    ["#0072B2", "#D55E00"]  # palette colorblind (bleu / orange)
):
    y_proba = modele.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{nom} (AUC = {auc:.3f})", color=couleur, linewidth=2)

# Ligne diagonale = référence du hasard
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Hasard (AUC = 0.5)")

plt.title("Courbes ROC — Comparaison des modèles", fontsize=13, fontweight="bold")
plt.xlabel("Taux de faux positifs")
plt.ylabel("Taux de vrais positifs (Recall)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("outputs/graphiques/courbes_roc.png", dpi=150)
print("Courbes ROC sauvegardées dans outputs/graphiques/courbes_roc.png")
plt.close()


# ============================================
# 4. COEFFICIENTS STANDARDISÉS DE LA RÉGRESSION LOGISTIQUE
# ============================================
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Standardisation des variables (mise à la même échelle)
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)

modele_logreg_std = LogisticRegression(C=10, random_state=42)
modele_logreg_std.fit(X_train_std, y_train)

coefficients = pd.DataFrame({
    "Variable": X.columns,
    "Coefficient": modele_logreg_std.coef_[0]
}).sort_values(by="Coefficient")

# Couleurs adaptées daltonisme : orange = augmente le risque, bleu = diminue
couleurs = ["#0072B2" if c < 0 else "#D55E00" for c in coefficients["Coefficient"]]

plt.figure(figsize=(7, 4))
plt.barh(coefficients["Variable"], coefficients["Coefficient"], color=couleurs)
plt.axvline(x=0, color="gray", linewidth=0.8)
plt.title("Effet de chaque variable sur le risque de churn\n(Régression logistique, coefficients standardisés)",
           fontsize=11, fontweight="bold")
plt.xlabel("Coefficient (négatif = diminue le risque, positif = l'augmente)")

for i, v in enumerate(coefficients["Coefficient"]):
    plt.text(v, i, f" {v:.3f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig("outputs/graphiques/coefficients_logreg.png", dpi=150)
print("\nGraphique des coefficients sauvegardé dans outputs/graphiques/coefficients_logreg.png")
plt.close()

print("\n=== Coefficients standardisés (comparables entre eux) ===")
print(coefficients)