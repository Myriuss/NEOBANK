
import pandas as pd

# Charger le dataset complet
df = pd.read_csv("data/application_train.csv")

# Créer les colonnes utiles pour le conseiller
df_subset = pd.DataFrame()
df_subset["SK_ID_CURR"] = df["SK_ID_CURR"]
df_subset["age"] = (-df["DAYS_BIRTH"] / 365).round(1)  # âge en années
df_subset["revenu"] = df["AMT_INCOME_TOTAL"]
df_subset["anciennete"] = (-df["DAYS_EMPLOYED"] / 365).replace({365243/365: 0}).round(1)  # gérer valeurs inconnues
df_subset["nb_incidents"] = df["DEF_30_CNT_SOCIAL_CIRCLE"].fillna(0) + df["DEF_60_CNT_SOCIAL_CIRCLE"].fillna(0)

# Ajouter un placeholder pour le score de crédit
df_subset["score_credit"] = None  # ou 0.0 en attendant le modèle

# Enregistrer ce dataset
df_subset.to_csv("dataset_conseiller.csv", index=False)