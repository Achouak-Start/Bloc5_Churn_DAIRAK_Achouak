import pandas as pd

# ============================================
# 1. CHARGEMENT DU DATASET BRUT
# ============================================
print("Chargement du dataset...")
df = pd.read_csv("data/raw/online_retail_II.csv", encoding="ISO-8859-1")

print(f"Nombre de lignes : {df.shape[0]}")
print(f"Nombre de colonnes : {df.shape[1]}")
print("\nAperçu des colonnes :")
print(df.dtypes)

# ============================================
# 2. INSPECTION INITIALE
# ============================================
print("\n=== Valeurs manquantes par colonne ===")
print(df.isnull().sum())

print("\n=== Statistiques descriptives ===")
print(df.describe())

# ============================================
# 3. NETTOYAGE
# ============================================

# Convertir la date en vrai format date
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Supprimer les lignes sans Customer ID (indispensable pour le churn :
# impossible de suivre un client anonyme dans le temps)
df = df.dropna(subset=["Customer ID"])

# Supprimer les lignes sans description produit
df = df.dropna(subset=["Description"])

# Supprimer les quantités négatives (retours/annulations)
df = df[df["Quantity"] > 0]

# Supprimer les prix invalides
df = df[df["Price"] > 0]

# Supprimer les doublons exacts
df = df.drop_duplicates()

# Créer la colonne montant total de la ligne
df["TotalAmount"] = df["Quantity"] * df["Price"]

print(f"\nAprès nettoyage : {df.shape[0]} lignes restantes")

# ============================================
# 4. VÉRIFICATION POST-NETTOYAGE
# ============================================
print("\n=== Vérification post-nettoyage ===")
print(f"Quantités négatives restantes : {(df['Quantity'] < 0).sum()}")
print(f"Prix <= 0 restants : {(df['Price'] <= 0).sum()}")
print(f"Customer ID manquants restants : {df['Customer ID'].isnull().sum()}")

# ============================================
# 5. SAUVEGARDE DU DATASET NETTOYÉ
# ============================================
df.to_csv("data/processed/online_retail_II_clean.csv", index=False)
print("\nDataset nettoyé sauvegardé dans data/processed/online_retail_II_clean.csv")