import pandas as pd

# ============================================
# 1. CHARGEMENT DU DATASET NETTOYÉ
# ============================================
df = pd.read_csv("data/processed/online_retail_II_clean.csv")
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# ============================================
# 2. DÉFINITION DES DEUX PÉRIODES
# ============================================
date_coupure = pd.Timestamp("2011-06-09")   # fin de la période d'observation
date_fin = df["InvoiceDate"].max()          # fin totale du dataset (~9 déc 2011)

periode_observation = df[df["InvoiceDate"] <= date_coupure]
periode_future = df[df["InvoiceDate"] > date_coupure]

print(f"Période d'observation : {periode_observation['InvoiceDate'].min()} -> {date_coupure}")
print(f"Période future        : {date_coupure} -> {date_fin}")
print(f"Clients dans la période d'observation : {periode_observation['Customer ID'].nunique()}")

# ============================================
# 3. CONSTRUCTION DES VARIABLES RFM (sur la période d'observation uniquement)
# ============================================
rfm = periode_observation.groupby("Customer ID").agg(
    Recence=("InvoiceDate", lambda x: (date_coupure - x.max()).days),
    Frequence=("Invoice", "nunique"),
    Montant=("TotalAmount", "sum")
).reset_index()

print("\nAperçu des variables RFM :")
print(rfm.head())
print(rfm.describe())

# ============================================
# 4. CONSTRUCTION DE LA VARIABLE CIBLE : CHURN
# ============================================
# Clients ayant racheté pendant la période future = non churnés
clients_actifs_futur = set(periode_future["Customer ID"].unique())

rfm["Churn"] = rfm["Customer ID"].apply(
    lambda cid: 0 if cid in clients_actifs_futur else 1
)

print("\nRépartition du churn :")
print(rfm["Churn"].value_counts())
print(rfm["Churn"].value_counts(normalize=True) * 100)

# ============================================
# 5. SAUVEGARDE
# ============================================
rfm.to_csv("data/processed/rfm_churn.csv", index=False)
print("\nJeu de données RFM + churn sauvegardé dans data/processed/rfm_churn.csv")